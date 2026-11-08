import numpy as np
import pytest
import soundfile as sf
from transcribe import read_audio


def test_audio_reader_downmixes_and_resamples(tmp_path):
    path=tmp_path/'stereo.wav'
    tone=np.sin(2*np.pi*440*np.arange(16000)/16000).astype(np.float32)
    sf.write(path,np.column_stack([tone,-tone]),16000,subtype='FLOAT')
    result=read_audio(path)
    assert result.shape==(8000,) and result.dtype==np.float32
    assert np.max(abs(result))<1e-7


def test_audio_reader_rejects_nonfinite_and_short(tmp_path):
    path=tmp_path/'bad.wav'
    sf.write(path,np.full(8000,np.nan,dtype=np.float32),8000,subtype='FLOAT')
    with pytest.raises(ValueError,match='finite'):read_audio(path)
    sf.write(path,np.zeros(50),8000)
    with pytest.raises(ValueError,match='third'):read_audio(path)
