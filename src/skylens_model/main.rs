//! Compute server. REST on 8100 (same endpoints as app.py), gRPC on 8101.

fn main() -> anyhow::Result<()> {
    skylens_model::app::main()
}
