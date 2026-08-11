# Demo Video — recording guide and word-for-word script

**Team BYTE BRIGADE · PS01 · SEMICON India Hackathon 2026**

Target length: **4 minutes 30 seconds** (limit is 5:00 — keep the margin).
Slide numbers below refer to **`BYTE BRIGADE_PS01.pdf`**, the submission deck.
*Italics* = what you say. **Bold** = what you do on screen.

---

## PART 1 — How to record (once, 15 minutes)

### Choose a recorder

**Easiest — Xbox Game Bar** (already on Windows): `Win + G`, then the record
button, or `Win + Alt + R` to start/stop. Captures screen + microphone;
files land in `Videos\Captures`. *Limitation: one window only, so you cannot
switch between the deck and the app mid-recording.*

**Recommended — OBS Studio** (free, obsproject.com): Sources → `+` →
**Display Capture** → OK, then Settings → Output → format `mp4`. Records the
whole screen, so you can move freely between the PDF and the app.

**Backup** — start a Zoom meeting alone, Share Screen, Record.

### Prepare the screen

1. Turn on **Focus Assist** so no notification appears mid-take.
2. Close anything you would not want on camera.
3. Open exactly two windows:
   - **`BYTE BRIGADE_PS01.pdf`** in full screen (`Ctrl + L` in most viewers)
   - The demo app:

     ```
     cd C:\Users\Mi\Downloads\files
     .venv\Scripts\streamlit.exe run app.py
     ```

4. In the app: sidebar → **"Restore a degraded image"**, and leave the
   dropdown on "(choose a sample)" so the restoration runs live on camera.
5. Test your mic: record 10 seconds and play it back.
6. **Do one full practice run before the real take.**

### Speaking tips

- Speak ~15% slower than feels natural; nerves speed you up.
- One narrator is cleaner than switching voices. If you split it, change
  speaker only at section boundaries.
- Fumbled a line? **Pause two seconds and say it again** — do not restart.

---

## PART 2 — The script

### SECTION 1 — Team and problem (0:00 – 0:55)

**Show: Slide 1 (Team Details)** — hold for about five seconds only.

> *Hello. We are Team BYTE BRIGADE from LNCT University, Bhopal — Prakhar,
> Raghav, Parth and Mayank — and this is our solution for problem statement
> PS01 from KLA: AI-Based Restoration of Degraded Images for Semiconductor
> Inspection.*

**Show: Slide 2 (Problem Statement Addressed)**

> *Inspection systems never capture perfectly clean images. To keep
> throughput high, images are captured quickly and at low dose, and that
> trades away quality.*

**Point at the picture at the bottom of the slide**

> *This is exactly what happens. On the left is the clean 256 by 256 ground
> truth. Three degradations are applied — speckle noise, which scales with
> brightness; additive Gaussian sensor noise, which is uniform; and two-times
> downsampling — in an order the organisers do not disclose. On the right is
> what our model actually receives: a small, heavily grained 128 by 128 image.*
>
> *Real detail hides inside that noise. Our task is to recover the clean
> 256 by 256 original, judged on PSNR, SSIM, LPIPS and end-to-end speed.*

### SECTION 2 — Our solution (0:55 – 2:05)

**Show: Slide 3 (Idea Description)**

> *Our approach rests on three decisions.*
>
> *First, supervised learning on the 3,200 paired images KLA provided — no
> external datasets, and no pretrained restoration weights.*
>
> *Second, a single network performs denoising and two-times
> super-resolution jointly, rather than chaining two separate models. One
> pass means one set of weights to train and no intermediate image where
> errors from the first stage could be amplified by the second.*
>
> *Third, and most important: our model corrects, rather than generates.*

**Show: Slide 4 (Proposed Solution)** — the architecture diagram

> *Here is the architecture, NAFNet-SR. A three-by-three convolution lifts the
> grayscale image into forty-eight feature channels. Twenty-four NAFNet blocks
> then clean those features — and the NAFNet block is remarkably simple: layer
> normalisation, a depthwise convolution, a SimpleGate and channel attention.
> No transformer, and not even a ReLU. That is the paper's key finding: the
> complicated parts of other restoration models were not what made them good.
> A PixelShuffle layer then learns the two-times upscale.*

**Trace the blue skip arrow across the top of the diagram**

> *Now the most important design decision — this blue path. We add a plain
> bicubic upscale of the input directly to the network's output. That means
> the network only ever predicts a correction: the noise to remove and the
> edges to sharpen, not the whole image.*
>
> *This strongly constrains hallucination. The model is anchored to the real
> input rather than free to generate plausible-looking structure — and for
> inspection that matters, because an invented edge could be read as a defect,
> or could mask a real one.*
>
> *The network has only zero point five three million parameters — about two
> point three megabytes as a saved inference checkpoint.*

### SECTION 3 — Live demonstration (2:05 – 3:20) ← the most important part

**Switch to the Streamlit app**

> *This is our demo application running the trained model. The sidebar shows
> what it loaded: NAFNet-SR, two-times scale, forty-eight channels,
> twenty-four blocks. For this live demo we are running entirely on CPU — no
> GPU at all. I will come back to our GPU benchmark in a moment.*

**Select `000102.npy` from the dropdown**

> *I am selecting an image the model has never seen — a real degraded file
> from our held-out test set.*

**While it computes, about a second:**

> *The restoration is happening right now, on a normal laptop processor.*

**Point at each panel in turn**

> *On the left, the degraded input, shown with its true pixels — 128 by 128
> and heavily grained. In the middle, plain bicubic upscaling: that is what
> you get with no AI at all — bigger, but the noise is all still there. On the
> right, our restoration — 256 by 256, noise removed, structure recovered.*

**Point at the zoom row underneath**

> *Below, we zoom into the centre of each, where the difference is clearest:
> dense grain, dense grain, and clean.*

**Point at the metrics**

> *The app scores it automatically: against its ground truth this image
> reaches thirty-two point five nine decibels, restored in about a second on
> CPU — ten and a half decibels better than the bicubic baseline, which gets
> twenty-two point zero six.*

**Select `000122.npy`**

> *Another unseen image, so you can see this is not cherry-picked — thirty
> point four nine decibels against bicubic's twenty-three point seven. In
> fact, our model beats the bicubic baseline on one hundred out of one hundred
> held-out test images.*

### SECTION 4 — Results (3:20 – 4:10)

**Show: Slide 6 (Impact and Benefits)** — the results strip and bar chart

> *For an honest evaluation we held out one hundred image pairs from the
> training data — never used in training or model selection — and scored all
> three judged metrics against their ground truth. The organisers' own hidden
> test set is separate; these are our own held-out pairs.*

**Point at the bar chart on the right of the slide**

> *PSNR rises from twenty-two point nine six to twenty-eight point zero zero
> decibels — a gain of five point zero four decibels, which corresponds to
> about a sixty-nine percent reduction in mean squared error.*
>
> *SSIM, structural similarity: zero point five three seven, up to zero point
> seven five nine.*
>
> *And LPIPS, the perceptual metric where lower is better: zero point four
> three three, down to zero point one seven five — sixty percent lower.*
>
> *Better on all three judged image-quality metrics.*
>
> *On speed: the live demo you just saw ran on CPU, but we also benchmarked on
> GPU. Measured end to end — disk read, transfers, the model itself and
> writing results — the same model reaches twenty-five milliseconds per image
> on an NVIDIA T4, or forty images per second. At that rate the four hundred
> official test images would take roughly ten seconds.*

### SECTION 5 — Honesty and close (4:10 – 4:30)

**Show: Slide 5 (Innovation and Uniqueness)**

> *We also analysed our failure cases. Performance drops on images with very
> dense fine texture, because information destroyed during downsampling
> cannot be reliably recovered, and our residual design deliberately avoids
> inventing that missing structure. Even in those difficult cases we still
> outperform bicubic — on all one hundred held-out images.*

**Show: Slide 8 (GitHub & Video Link)**

> *Everything is reproducible — training code, trained weights, the runtime
> report and sample outputs are all in our GitHub repository.*
>
> *Thank you. We are Team BYTE BRIGADE.*

---

## PART 3 — Every number, in one place

| Fact | Value |
|---|---|
| Team / college | BYTE BRIGADE, LNCT University, Bhopal |
| Problem statement | PS01, KLA |
| Input → output | 128×128 → 256×256 (2× scale) |
| Training pairs | 3,200 provided → 3,000 train / 100 val / 100 test |
| Model | NAFNet-SR, 0.53 M parameters |
| Checkpoint size | **2.3 MB** inference file (6.8 MB training file, which also stores optimizer state) |
| Blocks / channels | 24 NAFNet blocks, 48 channels |
| Training | 100 epochs, ~2 hours, free Colab T4 |
| Loss | Charbonnier + 0.15 SSIM + 0.05 LPIPS |
| Evaluated on | **our own 100 held-out pairs** (not the organisers' hidden set) |
| **PSNR** | **28.00 dB** (bicubic 22.96) |
| **SSIM** | **0.759** (bicubic 0.537) |
| **LPIPS** | **0.175** (bicubic 0.433 — lower is better) |
| Gain | +5.04 dB ≈ **69% reduction in MSE** |
| Speed — live demo | CPU, ~1.2 s/image |
| Speed — benchmark | **25 ms/image, 40 img/s on an NVIDIA T4**, end-to-end |
| 400 official test images | ~10 s at that rate (**extrapolated**, not measured on 400) |
| Demo image 000102 | ours 32.59 dB vs bicubic 22.06 dB |
| Demo image 000122 | ours 30.49 dB vs bicubic 23.68 dB |
| Win rate | 100 of 100 held-out images beat bicubic |

### Claims to state carefully

These are the sentences a technical judge is most likely to probe.

- **"Strongly constrained", not "impossible."** The residual skip anchors the
  output to the real input; it does not make hallucination mathematically
  impossible.
- **100 vs 400.** Our metrics come from *our own* 100 held-out pairs. The 400
  figure is the organisers' test set, and the ~10 s is an extrapolation from
  the measured 25 ms/image — say "would take", not "takes".
- **CPU vs GPU.** The live demo runs on CPU (~1.2 s/image); the 25 ms figure
  is the T4 benchmark. Always name which one you mean.
- **2.3 MB, not 6.5 MB.** 0.53 M float32 parameters is 2.1 MB of weights; the
  inference checkpoint is 2.3 MB. The 6.8 MB training file also carries AdamW
  optimizer state, which inference never uses.
- **MSE, not "pixel error".** +5.04 dB corresponds to a ~69% reduction in mean
  squared error (10^(−5.04/10) ≈ 0.31).
- **No joint-vs-chained ablation.** We did not run that experiment, so do not
  claim joint training "beats" a two-stage pipeline — say what it avoids
  instead.

---

## PART 4 — After recording

1. **Watch it once, end to end.** Audio audible? Live restoration visible?
2. **Check it is under 5:00.** If over, trim Section 2 — the most compressible.
3. **Upload as YouTube "Unlisted"** (or Drive → Anyone with the link).
4. **Test the link in an incognito window** — a Private video is invisible to
   judges even though it plays fine for you.
5. The link already sits on **Slide 8** of the deck; if you re-upload, replace
   it and re-export the PDF.

## Pre-upload checklist

- [ ] Under 5:00
- [ ] Audio clear throughout
- [ ] Live restoration visible on screen, not just slides
- [ ] PSNR, SSIM **and** LPIPS all spoken aloud
- [ ] GitHub repository shown or mentioned
- [ ] Link tested in an incognito window
