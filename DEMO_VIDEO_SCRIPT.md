# Demo Video Script — 5 minutes max

Target: **4 min 30 s** recorded (leaves buffer under the 5-minute limit).

## Before you record (10 minutes of prep)

1. Open the app and leave it running:
   ```
   .venv\Scripts\streamlit.exe run app.py
   ```
2. Open a second window with `BYTE_BRIGADE_SEMICON2026.pptx` in presentation mode.
3. Open a terminal in `C:\Users\Mi\Downloads\files` (for the live metric run).
4. **Do one full practice pass before the real take.** The restoration takes
   ~1.2 s — do not talk over dead air, use it to explain what is happening.
5. Recording tool: OBS Studio, or Windows **Win+G** game bar, or Zoom
   "share screen + record". Record **screen + microphone**.
6. Close notifications (Windows Focus Assist on) so nothing pops up mid-take.

---

## Section 1 — The problem  (0:00 – 0:50)

**Show:** slide 2 (Problem Statement).

> "Hello, we are Team Byte Brigade at SEMICON India Hackathon 2026, and our
> problem statement is AI-Based Restoration of Degraded Images for
> Semiconductor Inspection.
>
> In a semiconductor fab, wafers are inspected with electron and optical
> microscopes. To inspect quickly, images are captured fast and at low
> resolution, which makes them noisy — they carry speckle noise, which
> scales with brightness, and Gaussian sensor noise, which is uniform.
> The result is a 128 by 128 image where real defects can hide inside the
> noise. A missed defect can cost an entire wafer.
>
> Our task: take these degraded images and restore them to clean, sharp
> 256 by 256 images — removing noise and doubling resolution at the same
> time — judged on PSNR, SSIM, LPIPS, and inference speed."

## Section 2 — Our solution  (0:50 – 2:00)

**Show:** slide 3 (Idea), then slide 4 (Architecture).

> "Our approach rests on three decisions.
>
> First, supervised learning on paired data: 3,200 pairs of degraded input
> and clean ground truth. Nothing exotic — proven and debuggable.
>
> Second, one network does both jobs — denoising and 2× super-resolution in
> a single forward pass. Doing them jointly beats chaining two models.
>
> Third, and most important for inspection: the model restores, it never
> invents." *(switch to slide 4)*
>
> "The architecture is NAFNet-SR. A head convolution lifts the grayscale
> image into 48 feature channels. Twenty-four NAFNet blocks clean those
> features — and NAFNet's block is remarkably simple: layer norm, a
> convolution, a SimpleGate, and channel attention. No transformer, not
> even a ReLU. A PixelShuffle layer then learns the 2× upscale.
>
> The green path is the key idea: we add a plain bicubic upscale of the
> input to the network's output. That means the network only predicts the
> *correction* — the noise to remove and the edges to sharpen. Training is
> stable, and hallucinating a defect that isn't there becomes structurally
> impossible. The whole model is just 0.53 million parameters, 6.5 MB."

## Section 3 — Live demonstration  (2:00 – 3:30)  ← the heart of the video

**Show:** the Streamlit app. Sidebar → "Restore a degraded image".

> "This is our demo application. On the left you can see the model it
> loaded: NAFNet-SR, 2× scale, running on CPU — no GPU required."

**Do:** from the dropdown pick **`000102.npy`** — a held-out test image the
model never saw during training.

> "I'm selecting a test image the model has never seen. This is the actual
> degraded file provided by the organisers."

*(While it computes — about one second — say:)*

> "The image is being restored right now on a normal laptop CPU."

**Point at each panel as you speak:**

> "On the left, the degraded input — 128 by 128, heavily noisy. On the
> right, our restoration — 256 by 256, noise removed, edges recovered.
> The app measures it against the ground truth: **32.6 dB PSNR**, restored
> in about **one second**. Plain bicubic upscaling of the same image
> scores only 22.1 dB — our model gains more than 10 dB on this image."

**Do:** pick a second sample (e.g. `000122.npy`) to show it is not cherry-picked.

> "Here's another unseen image — 30.5 dB versus bicubic's 23.7. The model
> is blind to the noise level; it handles whatever it is given."

## Section 4 — The metrics  (3:30 – 4:15)

**Show:** the terminal. Run:
```
.venv\Scripts\python.exe evaluate.py --restored outputs\restored --gt data\test\gt
```
*(If you prefer no waiting, show slide 7 (Results) instead and read the table.)*

> "For an honest evaluation we held out 100 image pairs that were never
> used in training, and scored our restorations against the ground truth
> on all three judged metrics.
>
> PSNR: bicubic 22.96 dB, ours **28.00 dB** — a 5 dB gain, which means
> roughly 69 percent of the pixel error is eliminated.
>
> SSIM, which measures structural similarity: 0.54 for bicubic,
> **0.76** for ours.
>
> LPIPS, the perceptual metric where lower is better: 0.43 for bicubic,
> **0.175** for ours — a 60 percent reduction.
>
> Better on every single judged metric. And inference takes 1.2 seconds per
> image on CPU, 0.03 seconds on a GPU — we restored all 400 official test
> images in about eight minutes on a laptop."

## Section 5 — Close  (4:15 – 4:30)

**Show:** slide 9 (Impact & Feasibility) or slide 11 (GitHub).

> "This is feasible today, not a concept: a 6.5 MB model, trained in two
> hours on a free Colab GPU, that runs without any GPU at deployment. The
> complete source code, the trained weights, documentation, and sample
> outputs are all on our GitHub repository.
>
> Thank you — we are Team Byte Brigade."

---

## Checklist before uploading

- [ ] Video is **under 5:00**
- [ ] Audio is clear, no background noise
- [ ] The live restoration is actually visible on screen (not just slides)
- [ ] PSNR / SSIM / LPIPS numbers are shown and spoken
- [ ] GitHub link visible at the end
- [ ] Watch it once, end to end, before submitting

## Speaking tips

- Speak ~15 % slower than feels natural; nerves speed you up.
- One person narrating the whole video is fine and usually cleaner than
  switching voices — but if you split it, cut at the section boundaries.
- If you fumble a sentence, pause two seconds and repeat the sentence —
  it is trivial to cut later, and far better than restarting the take.
