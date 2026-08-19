// Where the core is, geographically.
//
// The control tower's route planner has to open its map SOMEWHERE. Opening it
// on a constant baked into the client is wrong the moment the system is
// deployed anywhere else, and asking the operator to pan there every time is
// worse. The core knows where it is better than the browser does — it is the
// machine sitting in the operations centre — so it resolves its own location
// once at startup and the tower asks for it.
//
// Three ways to answer, in order of trust:
//
//   config   SKYLENS_CORE_SITE="36.3685,127.3475"  — an operator said so
//   geoip    public IP lookup                       — a guess, city-accurate
//   fallback the configured anchor                  — nothing else worked
//
// The lookup is best-effort by design: a closed network (which is the real
// deployment) has no route to a geo-IP service, and the core must not spend its
// startup waiting for one to time out.

import type { Gps } from '../../shared/geo.ts';

export type SiteSource = 'config' | 'geoip' | 'fallback';

export interface Site {
  gps: Gps;
  source: SiteSource;
  /** City/region when the lookup gave one — shown so an operator can sanity
   *  check that the map opened where they expect. */
  label: string | null;
}

/** `lat,lon[,alt]` — the format SKYLENS_CORE_SITE takes. */
export function parseSite(raw: string | undefined): Gps | null {
  if (!raw) return null;
  const parts = raw.split(',').map((n) => Number(n.trim()));
  if (parts.length < 2 || parts.some((n) => !Number.isFinite(n))) {
    console.warn(`[core] SKYLENS_CORE_SITE="${raw}" is not lat,lon[,alt] — ignored`);
    return null;
  }
  const [lat, lon, alt] = parts;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    console.warn(`[core] SKYLENS_CORE_SITE="${raw}" is out of range — ignored`);
    return null;
  }
  return { lat, lon, alt: Number.isFinite(alt) ? alt : 0 };
}

interface GeoIpResponse {
  status?: string;
  lat?: number;
  lon?: number;
  city?: string;
  regionName?: string;
  country?: string;
}

/**
 * Resolve where this machine is. Never throws and never blocks longer than
 * `timeoutMs`: a site the operator can correct is better than a startup that
 * hangs waiting for the internet.
 */
export async function resolveSite(opts: {
  configured: string | undefined;
  fallback: Gps;
  lookup: boolean;
  lookupUrl: string;
  timeoutMs: number;
}): Promise<Site> {
  const configured = parseSite(opts.configured);
  if (configured) {
    return { gps: configured, source: 'config', label: null };
  }

  if (opts.lookup) {
    try {
      const res = await fetch(opts.lookupUrl, {
        signal: AbortSignal.timeout(opts.timeoutMs),
      });
      if (res.ok) {
        const body = (await res.json()) as GeoIpResponse;
        if (
          body.status !== 'fail' &&
          typeof body.lat === 'number' &&
          typeof body.lon === 'number'
        ) {
          const label = [body.city, body.regionName, body.country]
            .filter((s): s is string => typeof s === 'string' && s.length > 0)
            .join(', ');
          return {
            // Altitude is not something an IP knows; the fallback's is a better
            // guess than zero for prefilling a flight altitude.
            gps: { lat: body.lat, lon: body.lon, alt: opts.fallback.alt },
            source: 'geoip',
            label: label || null,
          };
        }
      }
      console.warn(`[core] site lookup answered ${res.status} — using the configured fallback`);
    } catch (err) {
      const why = err instanceof Error ? err.message : String(err);
      // Expected on a closed network. Say it once, plainly, and move on.
      console.warn(`[core] site lookup unavailable (${why}) — using the configured fallback`);
    }
  }

  return { gps: opts.fallback, source: 'fallback', label: null };
}
