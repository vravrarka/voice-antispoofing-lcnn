import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

class LogPowerSpectrogram(nn.Module):
    """
    Converts a waveform into a fixed-size log-power spectrogram.
    Input:
        waveform with shape [channels, samples]
    Output:
        spectrogram with shape [1, 863, 600]
    """
    def __init__(
        self,
        n_fft: int = 1724,
        win_length: int = 1724,
        hop_length: int = 130,
        target_frames: int = 600,
    ):
        super().__init__()
        self.target_frames = target_frames
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.blackman_window,
            power=2.0,
            normalized=False,
        )

    def forward(
        self,
        waveform: Tensor,
        random_crop: bool = False,
    ) -> Tensor:
        features = self.spectrogram(waveform)
        features = torch.log(
            features.clamp(min=1e-9)
        )
        num_frames = features.shape[-1]
        if num_frames < self.target_frames:
            padding = self.target_frames - num_frames
            features = F.pad(features, (0, padding))
        elif num_frames > self.target_frames:
            if random_crop:
                max_start = num_frames - self.target_frames
                start = torch.randint(
                    low=0,
                    high=max_start + 1,
                    size=(1,),
                ).item()
            else:
                start = 0
            features = features[
                ...,
                start:start + self.target_frames,
            ]
        return features
