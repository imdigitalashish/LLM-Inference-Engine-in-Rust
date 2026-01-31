use candle_core::Device;
use tracing::info;

use crate::error::Result;

pub fn get_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        info!("Using CPU device");
        return Ok(Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using CUDA device");
                return Ok(device);
            }
            Err(e) => {
                tracing::warn!("CUDA not available: {}, falling back to CPU", e);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Using Metal device");
                return Ok(device);
            }
            Err(e) => {
                tracing::warn!("Metal not available: {}, falling back to CPU", e);
            }
        }
    }

    info!("Using CPU device");
    Ok(Device::Cpu)
}

pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        Device::Metal(_) => "Metal".to_string(),
    }
}
