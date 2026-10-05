from pathlib import Path
import numpy as np
from front_end import MicFrontEnd

out = Path(__file__).resolve().parents[2] / "firmware" / "test"
rng = np.random.default_rng(123)
mic1 = (0.02 * rng.standard_normal(160)).astype(np.float32)
mic1[80] += 0.7
mic2 = (0.015 * rng.standard_normal(160)).astype(np.float32)
fe = MicFrontEnd()
y1, y2, trig = fe.process_frame(mic1, mic2)
mic1.tofile(out / "frontend_mic1_in.bin")
mic2.tofile(out / "frontend_mic2_in.bin")
y1.tofile(out / "frontend_expected_mic1.bin")
y2.tofile(out / "frontend_expected_mic2.bin")
(out / "frontend_expected_trigger.txt").write_text(str(int(trig)))
print('trigger', trig, 'max', float(np.max(np.abs(y1))))
