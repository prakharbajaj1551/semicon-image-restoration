# Demo Video — full recording guide and word-for-word script

**Team BYTE BRIGADE · PS01 · SEMICON India Hackathon 2026**

Target length: **4 minutes 30 seconds** (limit is 5:00 — leave the margin).
Everything in *italics* is what you say. Everything in **bold** is what you do.

---

## PART 1 — How to record (do this once, 15 minutes)

### Choose a recorder

**Easiest — Xbox Game Bar (already on Windows):** press `Win + G`, click the
record button (or `Win + Alt + R` to start/stop). It records the screen plus
your microphone. Files land in `Videos\Captures`.
*Limitation: it records one window at a time, so you cannot switch between
PowerPoint and the browser mid-recording.*

**Recommended — OBS Studio (free, obsproject.com):** records the whole
screen, so you can switch between slides and the app freely. Setup:
Sources → `+` → **Display Capture** → OK. Then Settings → Output → Recording
Quality "High", format `mp4`. Press **Start Recording**.

**Simplest alternative if both fail:** open a Zoom meeting alone, Share
Screen, and click Record. It produces an mp4 in your Documents\Zoom folder.

### Prepare your screen before recording

1. Turn on **Focus Assist** (Windows notifications off) —
   `Settings → System → Notifications → Focus assist → Alarms only`
2. Close Slack, WhatsApp, mail, and any tab you would not want on camera
3. Open **two windows** and nothing else:
   - `BYTE BRIGADE_PS01.pdf` (your submission deck) — press `Ctrl+L` in a
     PDF viewer for full screen
   - The demo app: open PowerShell, then

     ```
     cd C:\Users\Mi\Downloads\files
     .venv\Scripts\streamlit.exe run app.py
     ```

4. In the app: sidebar → **"Restore a degraded image"**. Leave the dropdown
   on "(choose a sample)" so the restoration happens live on camera.
5. Test your microphone: record 10 seconds, play it back. Check you are
   audible and there is no fan or echo.
6. **Do one full practice run before the real take.**

### Speaking tips

- Speak about 15% slower than feels natural — nerves speed everyone up.
- One narrator for the whole video is cleaner than switching voices. If you
  split it, change speakers only at section boundaries.
- If you fumble a sentence, **pause two seconds and say it again**. Do not
  restart the take — the pause makes it trivial to cut, and nobody expects
  a single perfect take.

---

## PART 2 — The script

### SECTION 1 — The problem (0:00 – 0:50)

**Show: Slide 2 of the PDF ("Problem Statement Addressed")**

> *Hello. We are Team BYTE BRIGADE from LNCT University, Bhopal, and this is
> our solution for problem statement PS01 from KLA — AI-Based Restoration of
> Degraded Images for Semiconductor Inspection.*
>
> *In a semiconductor fab, every wafer is inspected by microscope. To keep
> throughput high, those images are captured quickly and at low dose — and
> that trades away image quality. The result is a small, grainy image.*
>
> *There are three degradations. Speckle noise, which scales with brightness,
> so bright regions get noisier. Additive Gaussian sensor noise, which is the
> same everywhere. And two-times downsampling, which throws away resolution.
> KLA does not disclose the order they were applied in.*
>
> *The consequence is serious: a real defect can hide inside that noise. A
> missed defect can cost an entire wafer, and a false alarm wastes engineer
> time.*
>
> *Our task is to take a 128 by 128 degraded image and restore the clean
> 256 by 256 original — judged on PSNR, SSIM, LPIPS, and end-to-end speed.*

### SECTION 2 — Our solution (0:50 – 2:05)

**Show: Slide 3 ("Idea Description")**

> *Our approach rests on three decisions.*
>
> *First, supervised learning on the paired data KLA provided — three
> thousand two hundred pairs of degraded input and clean ground truth. No
> external datasets, and no pretrained restoration weights.*
>
> *Second, one single network does both jobs — denoising and two-times
> super-resolution in one forward pass. Doing them jointly beats chaining two
> separate models.*
>
> *Third, and most important for inspection: our model restores, it never
> invents.*

**Show: Slide 4 ("Proposed Solution")**

> *The architecture is NAFNet-SR. A three-by-three convolution lifts the
> grayscale image into forty-eight feature channels. Twenty-four NAFNet
> blocks then clean those features. NAFNet's block is remarkably simple —
> layer normalisation, a depthwise convolution, a SimpleGate, and channel
> attention. No transformer, and not even a ReLU activation. That is the
> paper's key finding: the complicated parts of other restoration models were
> not what made them good.*
>
> *A PixelShuffle layer then learns the two-times upscale, so new pixels are
> learned rather than interpolated.*
>
> *Now the most important design decision. We add a plain bicubic upscale of
> the input to the network's output. That means the network only ever
> predicts a correction — the noise to remove and the edges to sharpen. It
> makes training stable, and it makes hallucination structurally impossible.
> For defect inspection that matters enormously: a model that invents a
> plausible-looking edge could create a false defect, or hide a real one.*
>
> *The whole model is just zero point five three million parameters —
> six and a half megabytes.*

### SECTION 3 — Live demonstration (2:05 – 3:20) ← the most important part

**Switch to the Streamlit app**

> *This is our demo application, running the trained model. In the sidebar
> you can see what it loaded: NAFNet-SR, two-times scale, forty-eight
> channels, twenty-four blocks — and note it is running on CPU. No GPU is
> required.*

**Click the dropdown and select `000102.npy`**

> *I am now selecting a test image the model has never seen during training —
> this is a real degraded file from the held-out set.*

**While it computes — about one second — keep talking:**

> *The restoration is happening right now, on a normal laptop processor.*

**Point at each panel as you describe it**

> *On the left is the degraded input, shown with its true pixels — 128 by 128,
> heavily grained.*
>
> *In the middle is plain bicubic upscaling — that is what you get with no AI
> at all. It is bigger, but the noise is all still there.*
>
> *On the right is our restoration — 256 by 256, noise removed, and the
> underlying structure recovered.*

**Point at the zoom row underneath**

> *Underneath we zoom into the centre of each, where the difference is
> clearest — dense grain, dense grain, and clean.*

**Point at the metrics**

> *And the app scores it against the ground truth automatically: thirty-two
> point five nine decibels PSNR, restored in about a second, which is more
> than ten decibels better than the bicubic baseline on this image.*

**Select a second sample, `000122.npy`**

> *Here is another unseen image, to show this is not a cherry-picked example
> — thirty point four nine decibels, against bicubic's twenty-three point
> seven. In fact our model beats the bicubic baseline on one hundred out of
> one hundred held-out test images.*

### SECTION 4 — The metrics (3:20 – 4:10)

**Show: Slide 6 of the PDF ("Impact and Benefits")** — or run
`evaluate.py` live in the terminal if you prefer

> *For an honest evaluation we held out one hundred image pairs that were
> never used in training or in model selection, and scored all three judged
> metrics against the ground truth.*
>
> *PSNR: the bicubic baseline gets twenty-two point nine six decibels. Our
> model reaches twenty-eight point zero zero decibels. That is a gain of five
> point zero four decibels, which means roughly sixty-nine percent of the
> pixel error is eliminated.*
>
> *SSIM, which measures structural similarity: zero point five three seven
> for bicubic, zero point seven five nine for ours.*
>
> *And LPIPS, the perceptual metric where lower is better: zero point four
> three three for bicubic, zero point one seven five for ours — a sixty
> percent reduction.*
>
> *Better on every single judged metric.*
>
> *On speed: end-to-end — and that includes reading from disk, transfers to
> the GPU, the model itself, and writing the results — we measured twenty-five
> milliseconds per image on an NVIDIA T4, which is forty images per second.
> All four hundred official test images restore in about ten seconds.*

### SECTION 5 — Honesty and close (4:10 – 4:30)

**Show: Slide 5 ("Innovation and Uniqueness") or Slide 9 ("Research and
References")**

> *We also published our failure cases. Our gain correlates negatively with
> how much fine texture the ground truth already contains — when an image is
> densely textured, detail destroyed by downsampling can only be guessed at,
> and our design deliberately refuses to guess. Even in that worst case, we
> still beat the baseline.*
>
> *Everything is reproducible: the training code, the trained weights, the
> runtime report and the sample outputs are all in our GitHub repository,
> linked in this deck.*
>
> *Thank you. We are Team BYTE BRIGADE.*

---

## PART 3 — Every number you must say correctly

| Fact | Value |
|---|---|
| Team / college | BYTE BRIGADE, LNCT University, Bhopal |
| Problem statement | PS01, KLA |
| Input → output | 128×128 → 256×256 (2× scale) |
| Training pairs | 3,200 provided → 3,000 train / 100 val / 100 test |
| Model | NAFNet-SR, 0.53 M parameters, 6.5 MB |
| Blocks / channels | 24 NAFNet blocks, 48 channels |
| Training | 100 epochs, ~2 hours, free Colab T4 |
| Loss | Charbonnier + 0.15 SSIM + 0.05 LPIPS |
| **Our PSNR** | **28.00 dB** (baseline 22.96) |
| **Our SSIM** | **0.759** (baseline 0.537) |
| **Our LPIPS** | **0.175** (baseline 0.433, lower is better) |
| Gain | +5.04 dB, ≈69% of pixel error removed |
| Speed | 25 ms/image, 40 images/s on T4 (end-to-end) |
| All 400 test images | ~10 seconds |
| Demo image 000102 | ours 32.59 dB vs bicubic 22.06 dB |
| Demo image 000122 | ours 30.49 dB vs bicubic 23.68 dB |
| Win rate | beats bicubic on 100 of 100 held-out images |

---

## PART 4 — After recording

1. **Watch it once, all the way through.** Check the audio is audible and the
   live restoration is actually visible on screen.
2. **Check the length is under 5:00.** If over, trim Section 2 — it is the
   most compressible.
3. **Upload to YouTube as "Unlisted"** (recommended) or Google Drive.
   - YouTube: youtube.com → Create → Upload video → Visibility **Unlisted**
   - Drive: upload, then Share → General access → **Anyone with the link**
4. **Test the link in a private/incognito window** — if it does not play
   there, judges cannot watch it either.
5. **Send the link to Claude**, or paste it into slide 8 of the deck where it
   says `{PASTE VIDEO LINK HERE BEFORE SUBMITTING}`, then re-export to PDF.

## Pre-upload checklist

- [ ] Under 5:00
- [ ] Audio clear throughout
- [ ] Live restoration visible (not just slides)
- [ ] PSNR, SSIM **and** LPIPS all spoken aloud
- [ ] GitHub repository visible or mentioned
- [ ] Link tested in an incognito window
