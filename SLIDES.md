# PPT Content + Speaker Script — BYTE BRIGADE

(Slide-by-slide content matching BYTE_BRIGADE_SEMICON2026.pptx.
"Say:" lines are the 20-40 second speaker script per slide.)

---

## Slide 1 — Team
**BYTE BRIGADE** — AI-Based Restoration of Degraded Images for Semiconductor
Inspection — SEMICON India Hackathon 2026
Members: Prakhar Bajaj · Raghav Soni · Parth Kumar · Mayank Lodhi

> Say: "We are Byte Brigade. We built an AI system that takes noisy,
> low-resolution semiconductor inspection images and restores them to clean,
> sharp, double-resolution images — with one small, fast neural network."

## Slide 2 — Problem Statement
- Inspection images arrive **degraded**: speckle + Gaussian noise, and only
  128×128 resolution — defects hide in the noise
- Task: restore to clean 256×256, judged on **PSNR, SSIM, LPIPS + speed**
- Data: 3,200 paired images (degraded ↔ ground truth), 400 hidden test images

> Say: "In chip fabs, inspection images are noisy and low-resolution, and a
> hidden defect can cost a whole wafer. Our task: learn from 3,200 paired
> examples how to reverse the degradation, then restore 400 unseen test
> images that are scored against hidden ground truth."

## Slide 3 — Idea
1. **Learn from pairs** — supervised learning: degraded in, clean out
2. **One network, two jobs** — denoising + 2× super-resolution in a single
   forward pass (joint beats chained)
3. **Restore, never invent** — architecture and loss are designed so the
   model cannot hallucinate structure that isn't there

> Say: "Three principles. First, standard supervised learning on pairs — no
> exotic training. Second, one network does denoising and upscaling
> together, which works better than chaining two models. Third — critical
> for inspection — the model is built so it can only recover structure,
> never invent it. A fake defect or an erased one is worse than noise."

## Slide 4 — Architecture (NAFNet-SR)
Input 128×128 → Head conv (1→48 ch) → 24 × NAFBlock → PixelShuffle ×2 →
Tail conv → **add bicubic upscale of input** → Output 256×256
- NAFBlock = LayerNorm → conv → SimpleGate → channel attention (no ReLU!)
- Global skip: network learns only the *correction* on top of bicubic
- 0.53 M parameters · 6.8 MB file

> Say: "The backbone is NAFNet — ECCV 2022 state of the art. Its block is
> radically simple: no transformer, not even ReLU — a learned gate instead.
> Twenty-four blocks clean the features, PixelShuffle learns the upscale,
> and the key trick is the global skip: we add a plain bicubic upscale of
> the input, so the network only predicts the correction. That makes
> training stable and hallucination structurally impossible."

## Slide 5 — Innovation
- **Simplicity as strategy**: SOTA-family quality at 0.53 M params — we can
  explain and debug every line
- **Train on the scoreboard**: loss = Charbonnier + SSIM + light LPIPS —
  two of three judged metrics optimized directly
- **No-hallucination guarantee**: bicubic global skip + no GAN loss
- **Float-faithful pipeline**: native .npy I/O end to end — degraded inputs
  carry values outside [0,1]; we never clip them (PNG would)

> Say: "Our innovation is disciplined engineering, not exotic research. We
> train directly on the judged metrics. We guarantee no hallucination by
> construction. And one subtle point others may miss: the input arrays
> contain values outside zero-one — real noise signal. Our pipeline keeps
> full float precision everywhere; converting to images would destroy it."

## Slide 6 — Results
| | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic (no AI) | 22.96 dB | 0.537 | 0.433 |
| **Ours** | **28.00 dB** | **0.759** | **0.175** |

Callouts: **+5.0 dB** · **+0.22 SSIM** · **−60 % LPIPS** · **1.2 s/image (CPU)**
Comparison figure: degraded → bicubic → ours → ground truth

> Say: "On one hundred held-out pairs: plus five dB PSNR — that's roughly
> 69 percent of pixel error eliminated — SSIM up from 0.54 to 0.76, and
> LPIPS, the perceptual metric, cut by 60 percent. Better on every judged
> axis. And it runs at about a second per image on a plain laptop CPU —
> 30 milliseconds on a GPU."

## Slide 7 — Technology
- PyTorch · NAFNet-SR (ours, from scratch) · pytorch-msssim · LPIPS ·
  scikit-image · Streamlit demo UI
- Trained free: Google Colab T4, 100 epochs ≈ 2 h
- Reproducible: seeded splits, frozen benchmark, config stored inside the
  checkpoint — inference needs zero code edits

> Say: "Everything is open source and reproducible. Training cost nothing —
> two hours on a free Colab GPU. Every random seed is fixed, and the model
> file carries its own configuration, so anyone can rerun our exact
> pipeline from the README in four commands. There's also a live Streamlit
> demo — happy to show it."

## Slide 8 — GitHub
Repo: github.com/<add-link-here>  *(fill in once repo is created)*
- Modular: dataset/ · models/ · utils/ · train / evaluate / inference / app
- Every file commented with WHY, beginner-readable
- README: results, commands, and metric explanations

> Say: "The repository is small and readable on purpose — nine short
> modules, each commented to explain why, not just what. The README
> reproduces every number on this slide deck with four commands."

## Slide 9 — References
- Chen et al., "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022
- Shi et al., "Efficient Sub-Pixel CNN" (PixelShuffle), CVPR 2016
- Zhang et al., "The Unreasonable Effectiveness of Deep Features" (LPIPS), CVPR 2018
- PyTorch · pytorch-msssim · lpips · scikit-image · Streamlit

> Say: "We stand on proven open research — NAFNet for the architecture,
> PixelShuffle for upscaling, LPIPS for perceptual quality. Thank you —
> questions welcome."

---

### Anticipated Q&A (rehearse these)
- **Why not a GAN / diffusion model?** They hallucinate texture. In defect
  inspection, invented structure is a false positive; erased structure is a
  missed defect. Our fidelity-first loss and bicubic skip forbid both.
- **Why NAFNet over Restormer/SwinIR?** Same benchmark family, ~10× less
  complexity. The NAFNet paper showed the complicated parts weren't what
  made those models good. Smaller = faster inference (judged!) and fully
  explainable by us.
- **What is SimpleGate?** Split channels in half, multiply. One half acts
  as a learned gate for the other — the nonlinearity, with no ReLU needed.
- **How do you avoid overfitting?** Random crops + 8× flip/rotation
  augmentation, weight decay, and model selection on a held-out validation
  set the network never trains on.
- **Would it work on other noise levels?** Yes — trained across a range of
  degradations (blind restoration), and the demo accepts any input to prove it.
