/*
 * undistort.rs
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA implementation of `UndistortBackend`: the distorted-source lookup, bilinear warp and area-average downscale of
//! `CameraUndistortion::undistort_image` fused in one kernel. Output camera geometry still comes from the CPU code (computed once
//! per camera and cached), so only the pixel resampling moves to the device.

use crate::device::{CudaDevice, GpuError};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cumulus3d_core::{CameraModelKind, Error, Result};
use cumulus3d_dense::{CameraUndistortion, ImageBuffer, UndistortBackend};
use std::sync::{Arc, Mutex};

pub(crate) const SRC: &str = include_str!("kernels/undistort.cu");

/// CUDA image undistortion backend.
pub struct CudaUndistort {
    dev: Arc<CudaDevice>,
    f: CudaFunction,
    /// Only one use of the device at a time (stream shared). Image decoding stays parallel on the host.
    lock: Mutex<()>,
}

/// Kernel camera block: auxiliary pinhole (fx fy cx cy at input resolution) + source (fx fy cx cy k1 k2 p1 p2).
pub(crate) fn kernel_cameras(u: &CameraUndistortion) -> [f64; 12] {
    let src = &u.source;
    let mut aux = u.pinhole.clone();
    aux.rescale_to(src.width, src.height);
    let p = &src.params;
    let (k1, k2, p1, p2) = match src.model {
        CameraModelKind::SingleFocalPinhole | CameraModelKind::Pinhole => (0.0, 0.0, 0.0, 0.0),
        CameraModelKind::SingleFocalRadial => (p[3], 0.0, 0.0, 0.0),
        CameraModelKind::Radial => (p[3], p[4], 0.0, 0.0),
        CameraModelKind::OpenCv => (p[4], p[5], p[6], p[7]),
    };
    [
        aux.focal_length_x(),
        aux.focal_length_y(),
        aux.principal_point_x(),
        aux.principal_point_y(),
        src.focal_length_x(),
        src.focal_length_y(),
        src.principal_point_x(),
        src.principal_point_y(),
        k1,
        k2,
        p1,
        p2,
    ]
}

impl CudaUndistort {
    /// Compiles the undistortion kernel on device `dev`. FMA is off so rounding follows the CPU code.
    pub fn new(dev: Arc<CudaDevice>) -> std::result::Result<Self, GpuError> {
        let module = dev.compile(SRC, false, &[])?;
        Ok(Self { f: module.load_function("undistort_resample")?, dev, lock: Mutex::new(()) })
    }

    /// Builds with device 0.
    pub fn try_default() -> std::result::Result<Self, GpuError> {
        Self::new(CudaDevice::new(0)?)
    }

    /// GPU undistortion + downscale of one image. The input must have the source camera size and 1..=4 channels.
    pub fn undistort_gpu(&self, u: &CameraUndistortion, img: &ImageBuffer) -> std::result::Result<ImageBuffer, GpuError> {
        let cam = kernel_cameras(u);
        let (sw, sh, ch) = (img.width, img.height, img.channels);
        let (ow, oh) = (u.pinhole.width as usize, u.pinhole.height as usize);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let s = &self.dev.stream;
        let d_src = s.clone_htod(&img.data)?;
        let d_cam = s.clone_htod(&cam)?;
        let mut d_dst = s.alloc_zeros::<u8>(ow * oh * ch)?;
        let cfg =
            LaunchConfig { grid_dim: ((ow as u32).div_ceil(16), (oh as u32).div_ceil(16), 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 };
        let (sw_i, sh_i, ch_i, ow_i, oh_i) = (sw as i32, sh as i32, ch as i32, ow as i32, oh as i32);
        let mut l = s.launch_builder(&self.f);
        l.arg(&d_src).arg(&sw_i).arg(&sh_i).arg(&ch_i).arg(&mut d_dst).arg(&ow_i).arg(&oh_i).arg(&d_cam);
        // SAFETY: same arguments and sizes as undistort_resample(src[sw·sh·ch], sw, sh, ch, dst[ow·oh·ch], ow, oh, cam[12]).
        unsafe { l.launch(cfg) }?;
        let data = s.clone_dtoh(&d_dst)?;
        s.synchronize()?;
        Ok(ImageBuffer { width: ow, height: oh, channels: ch, data })
    }
}

impl UndistortBackend for CudaUndistort {
    fn undistort_image(&self, u: &CameraUndistortion, img: &ImageBuffer) -> Result<ImageBuffer> {
        if img.width != u.source.width as usize || img.height != u.source.height as usize {
            return Err(Error::InvalidArgument(format!(
                "undistort: image size {}x{} != camera {}x{}",
                img.width, img.height, u.source.width, u.source.height
            )));
        }
        if img.channels == 0 || img.channels > 4 {
            return u.undistort_image(img);
        }
        match self.undistort_gpu(u, img) {
            Ok(r) => Ok(r),
            Err(e) => {
                eprintln!("warning: CUDA undistortion failed ({e}), computing on CPU");
                u.undistort_image(img)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::Camera;
    use cumulus3d_dense::UndistortOptions;

    #[test]
    fn kernel_cameras_layout() {
        let cam = Camera::new(1, CameraModelKind::OpenCv, 200, 120, vec![160.0, 161.0, 100.0, 60.0, 0.05, -0.02, 0.001, 0.0005]).unwrap();
        let u = CameraUndistortion::new(&cam, &UndistortOptions::default()).unwrap();
        let k = kernel_cameras(&u);
        assert_eq!(&k[4..], &[160.0, 161.0, 100.0, 60.0, 0.05, -0.02, 0.001, 0.0005]);
        assert!(k[..4].iter().all(|&v| v > 0.0));
        let r = Camera::new(2, CameraModelKind::SingleFocalRadial, 200, 120, vec![160.0, 100.0, 60.0, 0.07]).unwrap();
        let u = CameraUndistortion::new(&r, &UndistortOptions::default()).unwrap();
        assert_eq!(&kernel_cameras(&u)[4..], &[160.0, 160.0, 100.0, 60.0, 0.07, 0.0, 0.0, 0.0]);
    }
}
