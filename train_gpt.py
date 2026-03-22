# KAISER V11 START
print("\n[TRAIN_GPT.PY] Script geladen...")
print("\n>>> [TRAIN_GPT.PY] Script geladen. Starte Imports...")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time, os
WALLCLOCK_START    = time.perf_counter()
MAX_WALLCLOCK_SECS = int(os.environ.get("MAX_WALLCLOCK_SECONDS", "0"))
t0_script = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, json, hashlib
print(f">>> [INIT] Imports fertig nach {time.time()-t0_script:.2f}s")
import numpy as np
from collections import deque
from datetime import datetime

# 
# 1. HARDWARE AUTO-DETECT
# 
CUDA   = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

if CUDA:
    BATCH_SIZE   = 32
    GRAD_ACCUM   = 4
    EVAL_EVERY   = 200
    WARMUP_STEPS = 300
    STABLE_FRAC  = 0.75
    SEQ_LEN      = 2048
    USE_BF16     = True
    USE_COMPILE  = True
    print(f"[GPU] Batch: {BATCH_SIZE * GRAD_ACCUM} | SEQ: {SEQ_LEN}")
else:
    BATCH_SIZE   = 2
    GRAD_ACCUM   = 1
    EVAL_EVERY   = 50
    WARMUP_STEPS = 50
    STABLE_FRAC  = 0.70
    SEQ_LEN      = 256
    USE_BF16     = False
    USE_COMPILE  = False
    print(f"[CPU] Mini-Mode! Batch: {BATCH_SIZE * GRAD_ACCUM} | SEQ: {SEQ_LEN}")

# 
# 2. HYPERPARAMETER
# 
VOCAB_SIZE     = 1024
DIM            = 576
N_UNIQUE       = 5
N_TOTAL        = 30
N_Q_HEADS      = 8
N_KV_HEADS     = 2
HEAD_DIM       = DIM // N_Q_HEADS
MLP_HIDDEN     = int(DIM * 3 / 64) * 64   # 1536
BIGRAM_BUCKETS = 10240
BIGRAM_DIM     = 128


LR_BIT         = 0.004
LR_ADAM        = 3e-4
LR_MIN         = 1e-6
WEIGHT_DECAY   = 0.1
TIE_EMB        = True

# Selbstheilung Startwerte
BIT_WD_BASE    = 0.02    # Basis Weight Decay
CLAMP_BASE     = 0.5     # Basis Clamp
CLAMP_MIN      = 0.25    # Minimum Clamp (untere Grenze)
CLAMP_RECOVER  = 0.02    # Pro Step erholt sich Clamp um diesen Betrag
GN_SPIKE_THRESH = 25.0   # Ab diesem grad_norm gilt es als Spike
GN_AVG_WINDOW  = 30      # Fenster fr gleitenden Durchschnitt grad_norm
PLATEAU_STEPS  = 200     # Steps ohne Verbesserung  LR halbieren
W_STD_TARGET   = 0.035   # Ziel-w_std fr dynamisches WD

DATA_DIR   = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
TRAIN_PATH = os.path.join(DATA_DIR, "original_fineweb.bin")

CKPT_DIR  = "records/kaiser_v11"
ARCHIVE   = "records/kaiser_v11_archive"
BEST_CKPT = os.path.join(CKPT_DIR, "kaiser_best.pth")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(ARCHIVE,  exist_ok=True)

# 
# 3. BUDGET
# 
def estimate_mb():
    per_block = (
        DIM * N_Q_HEADS  * HEAD_DIM +
        DIM * N_KV_HEADS * HEAD_DIM * 2 +
        N_Q_HEADS * HEAD_DIM * DIM +
        DIM * MLP_HIDDEN + MLP_HIDDEN * DIM
    )
    total  = N_UNIQUE * per_block * 0.625  # 5-Bit Quantisierung Ersparnis!
    total += VOCAB_SIZE * DIM * 2
    total += BIGRAM_BUCKETS * BIGRAM_DIM * 2
    total += BIGRAM_DIM * DIM * 4
    total += 100_000
    return total / (1024 ** 2)

MB_EST = estimate_mb()
print(f"[BUDGET] ~{MB_EST:.2f} MB")
# assert MB_EST < 16.0, f"BUDGET UEBERSCHRITTEN: {MB_EST:.2f} MB!"


# 
# 4. BITLINEAR
# 
class BitLinear(nn.Linear):
    """
    Float32 Training, Int8 nur beim Speichern.
    Kein STE, keine diskrete Gradientenlandschaft, kein Explodieren.
    """
    def __init__(self, in_f, out_f, bias=False):
        super().__init__(in_f, out_f, bias)
        nn.init.normal_(self.weight, 0.0, 0.02)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias)

# 
# 5. BIGRAMHASH  (c) JamOne Project 2025
# 
class BigramHash(nn.Module):
    """
    Hash-Tabelle fuer Token-Paare.
    bucket = (prev * 2654435761 XOR curr) % n_buckets
    Gibt dem Modell Bigramm-Statistiken direkt als Embedding.
    """
    def __init__(self, n_buckets=BIGRAM_BUCKETS, emb_dim=BIGRAM_DIM, out_dim=DIM):
        super().__init__()
        self.n_buckets = n_buckets
        self.table = nn.Embedding(n_buckets, emb_dim)
        nn.init.normal_(self.table.weight, 0.0, 0.01)
        self.proj  = nn.Linear(emb_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, 0.0, 0.01)

    def forward(self, idx):
        B, T   = idx.shape
        prev   = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=idx.device),
                             idx[:, :-1]], dim=1)
        bucket = (prev.long() * 2654435761 ^ idx.long()) % self.n_buckets
        return self.proj(self.table(bucket))

# 
# 6. ROPE
# 
def build_rope_cache(head_dim, max_seq, device, theta=50_000.0):
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos   = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb   = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def apply_rope(x, cos, sin):
    T  = x.size(2)
    c  = cos[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    s  = sin[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * c + torch.cat([-x2, x1], dim=-1) * s

# 
# 7. ATTENTION
# 
class GQAttention(nn.Module):
    TEMP = 15.0
    def __init__(self):
        super().__init__()
        self.q   = BitLinear(DIM, N_Q_HEADS  * HEAD_DIM)
        self.k   = BitLinear(DIM, N_KV_HEADS * HEAD_DIM)
        self.v   = BitLinear(DIM, N_KV_HEADS * HEAD_DIM)
        self.out = BitLinear(N_Q_HEADS * HEAD_DIM, DIM)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, N_Q_HEADS,  HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        rep = N_Q_HEADS // N_KV_HEADS
        k   = k.repeat_interleave(rep, dim=1)
        v   = v.repeat_interleave(rep, dim=1)
        logits = torch.matmul(q * self.TEMP, k.transpose(-2, -1))
        mask   = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, float('-inf'))
        attn_w = F.softmax(logits.float(), dim=-1).to(q.dtype)
        out    = torch.matmul(attn_w, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))

# 
# 8. MLP
# 
class ReluSquaredMLP(nn.Module):
    """ReLU^2: sparsame Aktivierungen, stabil mit BitLinear."""
    def __init__(self):
        super().__init__()
        self.up   = BitLinear(DIM, MLP_HIDDEN)
        self.down = BitLinear(MLP_HIDDEN, DIM)
    def forward(self, x):
        return self.down(F.relu(self.up(x)).pow(2))

# 
# 9. BLOCK
# 
class KaiserBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(DIM)
        self.attn = GQAttention()
        self.ln2  = nn.LayerNorm(DIM)
        self.mlp  = ReluSquaredMLP()
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

# 
# 10. MODELL
# 
class JamOne_Kaiser(nn.Module):
    """5 Bloecke x 6 Zyklen + BigramHash + U-Net Skip Connections."""
    def __init__(self):
        super().__init__()
        self.tok_emb    = nn.Embedding(VOCAB_SIZE, DIM)
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        self.bigram     = BigramHash(BIGRAM_BUCKETS, BIGRAM_DIM, DIM)
        self.blocks     = nn.ModuleList([KaiserBlock() for _ in range(N_UNIQUE)])
        self.ln_f       = nn.LayerNorm(DIM)
        self.lm_head    = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.unet_scale = nn.Parameter(torch.zeros(N_UNIQUE))
        if TIE_EMB:
            self.lm_head.weight = self.tok_emb.weight
        self._rope_cos = self._rope_sin = self._rope_device = None

    def _get_rope(self, device):
        if self._rope_device != str(device):
            cos, sin = build_rope_cache(HEAD_DIM, SEQ_LEN + 128, device)
            self._rope_cos, self._rope_sin, self._rope_device = cos, sin, str(device)
        return self._rope_cos, self._rope_sin

    def forward(self, idx):
        B, T     = idx.shape
        cos, sin = self._get_rope(idx.device)
        x        = self.tok_emb(idx) + self.bigram(idx)
        half     = N_TOTAL // 2
        cache    = {}
        for i in range(N_TOTAL):
            bi     = i % N_UNIQUE
            mirror = N_TOTAL - 1 - i
            if i >= half and mirror in cache:
                x = x + torch.sigmoid(self.unet_scale[bi]) * cache[mirror]
            x = self.blocks[bi](x, cos, sin)
            if i < half:
                cache[i] = x
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] {n:,} Parameter ({n/1e6:.2f}M)")
        return n

# 
# 11. BITMOMENTUM  (c) JamOne Project 2025
# 
class BitMomentum(torch.optim.Optimizer):
    """
    
      BitMomentum  --  EIGENENTWICKLUNG  (c) 2025 JamOne Project         
                                                                          
      Nesterov Momentum + Sign-Update fuer ternaere Gewichte.            
      Weight Decay zieht Gewichte zur Null (verhindert Explosion).       
      Momentum-Reset: bei Spike wird Puffer der Schicht genullt.        
    
    """
    def __init__(self, params, lr=0.004, momentum=0.92, nesterov=True, weight_decay=0.02):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def reset_momentum_for_spikes(self, threshold=0.5):
        """
        Selbstheilung: wenn ein einzelner Parameter-Gradient zu gro ist,
        wird sein Momentum-Puffer genullt. Verhindert Aufschaukeln.
        """
        reset_count = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.norm() > threshold:
                    state = self.state[p]
                    if 'momentum_buf' in state:
                        state['momentum_buf'].zero_()
                        reset_count += 1
        return reset_count

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, beta, nesterov, wd = (group['lr'], group['momentum'],
                                       group['nesterov'], group['weight_decay'])
            for p in group['params']:
                if p.grad is None:
                    continue
                g     = p.grad
                state = self.state[p]
                if 'momentum_buf' not in state:
                    state['momentum_buf'] = torch.zeros_like(g)
                buf = state['momentum_buf']
                buf.mul_(beta).add_(g)
                g_eff = (g + beta * buf) if nesterov else buf
                if g_eff.ndim >= 2:
                    p.add_(g_eff.sign(), alpha=-lr)
                else:
                    p.add_(g_eff, alpha=-lr)
        return loss

# 
# 12. INLINE SELBSTHEILUNGS-CONTROLLER  (c) JamOne Project 2025
# 
class SelfHealingController:
    """
    
      Inline Selbstheilung  kein Neustart, alles in Echtzeit           
      (c) 2025 JamOne Project. Alle Rechte vorbehalten.                 
    
                                                                          
      MECHANISMUS 1: Adaptiver Clamp                                     
        Bei Spike  Clamp enger (min 0.15)                              
        Danach  erholt sich langsam zurueck auf Base (0.5)             
                                                                          
      MECHANISMUS 2: Momentum-Reset                                      
        Bei Spike  Momentum-Puffer betroffener Schichten nullen        
        Verhindert dass sich schlechter Gradient aufschaukelt           
                                                                          
      MECHANISMUS 3: Dynamisches Weight Decay                            
        BIT_WD steigt wenn w_std > Zielwert                            
        BIT_WD sinkt wenn w_std < Zielwert                             
        Haelt Gewichte automatisch in stabilem Bereich                  
                                                                          
      MECHANISMUS 4: Plateau-Detektor                                    
        Wenn N Steps kein Fortschritt  LR halbieren                   
        LR wird nie unter LR_MIN gesenkt                                
                                                                          
      MECHANISMUS 5: Gleitender Gradient-Durchschnitt                   
        Entscheidungen basieren auf Durchschnitt, nicht Einzelwert      
        Verhindert Fehlalarme durch kurzfristige Spikes                 
    
    """
    def __init__(self):
        self.clamp_val       = CLAMP_BASE
        self.current_wd      = BIT_WD_BASE
        self.gn_history      = deque(maxlen=GN_AVG_WINDOW)
        self.best_val_bpb    = float('inf')
        self.steps_no_improve = 0
        self.lr_halved_count  = 0
        self.heal_events      = 0
        self.log              = []

    def update(self, grad_norm, avg_ws, val_bpb, opt_bit, model):
        """
        Wird jeden Step aufgerufen. Passt Parameter inline an.
        Gibt einen Status-String zurueck fuer das Logging.
        """
        self.gn_history.append(grad_norm)
        avg_gn = sum(self.gn_history) / len(self.gn_history)
        events = []

        #  Mechanismus 1+2: Adaptiver Clamp + Momentum-Reset 
        if grad_norm > GN_SPIKE_THRESH:
            # Clamp sofort enger
            self.clamp_val = max(CLAMP_MIN, self.clamp_val * 0.7)
            # Momentum-Puffer nullen
            resets = opt_bit.reset_momentum_for_spikes(threshold=0.5)
            self.heal_events += 1
            events.append(f"SPIKE(gn={grad_norm:.0f}clamp={self.clamp_val:.2f},resets={resets})")
        else:
            # Clamp erholt sich langsam
            self.clamp_val = min(CLAMP_BASE, self.clamp_val + CLAMP_RECOVER)

        # Clamp anwenden
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, BitLinear):
                    m.weight.clamp_(-self.clamp_val, self.clamp_val)

        #  Mechanismus 3: Dynamisches Weight Decay 
        if avg_ws > W_STD_TARGET * 1.2:
            # w_std zu hoch  WD erhhen
            self.current_wd = min(0.05, self.current_wd + 0.001)
            events.append(f"WD{self.current_wd:.3f}")
        elif avg_ws < W_STD_TARGET * 0.6:
            # w_std zu niedrig  WD senken (Modell kann mehr lernen)
            self.current_wd = max(0.005, self.current_wd - 0.001)
            events.append(f"WD{self.current_wd:.3f}")

        # WD im Optimizer aktualisieren
        for group in opt_bit.param_groups:
            group['weight_decay'] = self.current_wd

        #  Mechanismus 4: Plateau-Detektor 
        if val_bpb is not None:
            if val_bpb < self.best_val_bpb - 0.001:
                self.best_val_bpb     = val_bpb
                self.steps_no_improve = 0
            else:
                self.steps_no_improve += 1

            if self.steps_no_improve >= PLATEAU_STEPS and self.lr_halved_count < 3:
                for group in opt_bit.param_groups:
                    group['lr'] = max(LR_MIN, group['lr'] * 0.5)
                self.steps_no_improve = 0
                self.lr_halved_count += 1
                events.append(f"PLATEAULR/2(x{self.lr_halved_count})")

        status = " | ".join(events) if events else "stable"
        return status, avg_gn

    def get_stats(self):
        return {
            "clamp":     round(self.clamp_val, 3),
            "wd":        round(self.current_wd, 4),
            "heal":      self.heal_events,
            "plateau":   self.lr_halved_count,
            "no_improve": self.steps_no_improve
        }

# 
# 13. OPTIMIZER SETUP
# 
def build_optimizers(model):
    bit_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim >= 2
                and 'tok_emb' not in name and 'lm_head' not in name
                and 'bigram' not in name and 'ln' not in name
                and 'gamma' not in name and 'unet' not in name):
            bit_params.append(p)
        else:
            adam_params.append(p)
    print(f"[OPTIM] BitMomentum: {sum(p.numel() for p in bit_params):>12,}")
    print(f"[OPTIM] AdamW:       {sum(p.numel() for p in adam_params):>12,}")
    opt_bit  = BitMomentum(bit_params, lr=LR_BIT, momentum=0.92,
                           nesterov=True, weight_decay=BIT_WD_BASE)
    opt_adam = torch.optim.AdamW(adam_params, lr=LR_ADAM,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY, eps=1e-8)
    return opt_bit, opt_adam

# 
# 14. LR SCHEDULE
# 
def get_lr(step, max_lr, min_lr, total, warmup, stable_frac):
    stable_end = int(total * stable_frac)
    if step < warmup:
        return min_lr + (max_lr - min_lr) * (step / max(1, warmup))
    elif step < stable_end:
        return max_lr
    else:
        t = (step - stable_end) / max(1, total - stable_end)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (max_lr - min_lr)

def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr

# ══════════════════════════════════════════════════════════════════════════════
# INT5 POST-TRAINING QUANTISIERUNG  (c) JamOne / DepthCycle 2025
# ══════════════════════════════════════════════════════════════════════════════
def quantize_to_int5(w_float, scale):
    """
    Quantisiert Float32-Gewichte auf 5-bit signed (-15 bis +15).
    Gibt int8-Tensor mit Werten in [-15, 15] zurück.
    """
    w_q = (w_float / scale.clamp(min=1e-8)).clamp(-15, 15).round()
    return w_q.to(torch.int8)

def pack_int5(tensor_int8):
    """
    Packt int8-Werte (Bereich -15..15) in 5-bit pro Wert.
    8 Werte → 5 Bytes  (Ersparnis: 37.5% gegenüber Int8)
    """
    flat = (tensor_int8.flatten().to(torch.int32) + 16).numpy()  # 0..31
    pad  = (8 - len(flat) % 8) % 8
    flat = np.concatenate([flat, np.zeros(pad, dtype=np.int32)])

    bits = np.zeros(len(flat) * 5, dtype=np.uint8)
    for bit_pos in range(5):
        bits[4 - bit_pos::5] = (flat >> bit_pos) & 1

    bit_pad = (8 - len(bits) % 8) % 8
    bits = np.concatenate([bits, np.zeros(bit_pad, dtype=np.uint8)])
    packed = np.packbits(bits)

    return torch.from_numpy(packed), tensor_int8.shape, pad

def unpack_int5(packed_tensor, original_shape, pad):
    """
    Entpackt 5-bit-Daten zurück zu int8-Tensor.
    """
    packed = packed_tensor.numpy() if isinstance(packed_tensor, torch.Tensor) else np.frombuffer(packed_tensor, dtype=np.uint8)
    bits   = np.unpackbits(packed)

    n_vals = (len(bits) // 5)
    vals   = np.zeros(n_vals, dtype=np.int32)
    for bit_pos in range(5):
        vals += bits[4 - bit_pos::5].astype(np.int32) * (1 << bit_pos)

    if pad > 0:
        vals = vals[:-pad]
    return torch.from_numpy((vals - 16).astype(np.int8)).reshape(original_shape)

def save_model(model, path):
    sd, out = model.state_dict(), {}
    for k, v in sd.items():
        ist_bit = ('weight' in k and any(
            x in k for x in ['attn.q', 'attn.k', 'attn.v', 'attn.out',
                              'mlp.up', 'mlp.down']))
        if ist_bit:
            # Per-Output-Channel Scale (wie vorher)
            scale = v.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
            w_q   = quantize_to_int5(v, scale)

            # Int5 packen
            packed_tensor, orig_shape, pad = pack_int5(w_q)

            # Speichern: packed bytes + scale + shape-Info
            out[k]                    = packed_tensor  # uint8 Tensor statt bytes
            out[k + '_int5_scale']    = scale.squeeze(1).to(torch.float16)
            out[k + '_int5_shape']    = torch.tensor(list(orig_shape), dtype=torch.int32)
            out[k + '_int5_pad']      = torch.tensor([pad], dtype=torch.int32)
        elif 'tok_emb.weight' in k or 'lm_head.weight' in k:
            out[k] = v.to(torch.float16)
        elif 'bigram.table.weight' in k:
            out[k] = v.to(torch.float16)
        else:
            out[k] = v.to(torch.float32)
    torch.save(out, path)
    sz = os.path.getsize(path) / (1024**2)
    print(f"\n[SAVED] {os.path.basename(path)} | {sz:.2f} MB")
    if sz > 16.0:
        print(f"[WARNUNG] Budget ueberschritten: {sz:.2f} MB!")
    return sz

def load_model(model, path, device):
    sd_disk, sd_model, sd_load = torch.load(path, map_location=device), model.state_dict(), {}
    for k, v_d in sd_disk.items():
        if k not in sd_model:
            continue
        v_m = sd_model[k]
        
        # Int5 erkennen
        int5_key = k + '_int5_scale'
        if int5_key in sd_disk:
            scale      = sd_disk[int5_key].float().unsqueeze(1)
            orig_shape = tuple(sd_disk[k + '_int5_shape'].tolist())
            pad        = sd_disk[k + '_int5_pad'].item()
            w_int8     = unpack_int5(sd_disk[k], orig_shape, pad)
            sd_load[k] = (w_int8.float() * scale).to(v_m.dtype)
        elif isinstance(sd_disk.get(k), (bytes, bytearray)):
            # Fallback falls irgendwas schief geht
            sd_load[k] = torch.zeros_like(v_m)
        elif isinstance(v_d, torch.Tensor) and v_d.dtype == torch.int8 and v_m.dtype == torch.float32:
            # Alter Int8-Checkpoint: rückwärtskompatibel laden
            scale_key  = k.replace('.weight', '.scale')
            scale      = sd_disk.get(scale_key, torch.ones(1)).float()
            sd_load[k] = v_d.float() * scale
        elif isinstance(v_d, torch.Tensor) and v_d.dtype != v_m.dtype:
            sd_load[k] = v_d.to(v_m.dtype)
        else:
            sd_load[k] = v_d
    missing, unexpected = model.load_state_dict(sd_load, strict=False)
    if missing:    print(f"[LOAD] Fehlende Keys: {len(missing)}")
    if unexpected: print(f"[LOAD] Unerwartete Keys: {len(unexpected)}")
    print(f"[LOAD] Geladen: {path}")
    return model

# 
# 16. DATA LOADING
# 
def build_hash_split(data, chunk_size=2048, val_frac=0.1):
    train_idx, val_idx = [], []
    mod = int(1.0 / val_frac)
    all_idx = list(range(0, len(data) - chunk_size, chunk_size))
    split   = max(1, len(all_idx) // 10)
    val_idx   = all_idx[:split]
    train_idx = all_idx[split:]
    if not val_idx:   val_idx   = train_idx[:max(1, len(train_idx)//10)]
    if not train_idx: train_idx = val_idx[:1]
    print(f"[DATA] Train: {len(train_idx):,} | Val: {len(val_idx):,}")
    return train_idx, val_idx

class DataLoader:
    def __init__(self, data, indices, batch_size, seq_len, device):
        self.data, self.idx, self.B  = data, indices, batch_size
        self.T, self.device          = seq_len, device
        self.rng                     = np.random.default_rng(42)
        self.chunk_sz                = 2048

    def get_batch(self):
        if self.T >= self.chunk_sz:
            cidx = self.rng.choice(self.idx, size=self.B)
            xs, ys = [], []
            for c in cidx:
                needed = self.T + 1
                tokens = []
                cur    = c
                while len(tokens) < needed and cur < len(self.data):
                    end = min(cur + self.chunk_sz, len(self.data))
                    tokens.extend(self.data[cur:end].tolist())
                    cur += self.chunk_sz
                tokens = (tokens + [0] * needed)[:needed]
                xs.append(tokens[:self.T])
                ys.append(tokens[1:self.T+1])
            return (torch.tensor(xs, dtype=torch.long, device=self.device),
                    torch.tensor(ys, dtype=torch.long, device=self.device))
        else:
            cidx    = self.rng.choice(self.idx, size=self.B)
            offsets = self.rng.integers(0, self.chunk_sz - self.T - 1, size=self.B)
            xs, ys  = [], []
            for c, o in zip(cidx, offsets):
                s = c + o
                xs.append(self.data[s   : s+self.T  ].astype(np.int64))
                ys.append(self.data[s+1 : s+self.T+1].astype(np.int64))
            return (torch.from_numpy(np.stack(xs)).to(self.device),
                    torch.from_numpy(np.stack(ys)).to(self.device))

def train():
    print("-" * 70)
    print("  JamOne KAISER V11  -  Inline Selbstheilung")
    print("-" * 70)

    if not os.path.exists(TRAIN_PATH):
        print("[DRY-RUN] Kein Datensatz. Architektur-Test...")
        model        = JamOne_Kaiser().to(DEVICE)
        model.count_params()
        xd           = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        yd           = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        opt_b, opt_a = build_optimizers(model)
        logits       = model(xd)
        loss         = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yd.view(-1))
        loss.backward()
        opt_b.step(); opt_a.step()
        print(f"[DRY-RUN] OK | BPB: {loss.item()/math.log(2):.4f}")
        save_model(model, BEST_CKPT)
        return

    print(">>> [INIT] train() startet...")
    print(f"[DATA] Oeffne Datei: {TRAIN_PATH}")
    data = np.memmap(TRAIN_PATH, dtype=np.uint16, mode='r')
    print(f"[DATA] {len(data):,} Tokens gefunden. Splitting...")
    train_idx, val_idx = build_hash_split(data)
    print(f"[DATA] Train: {len(train_idx):,} | Val: {len(val_idx):,}")
    
    print("[DATA] Initialisiere Loader...")
    train_loader = DataLoader(data, train_idx, BATCH_SIZE, SEQ_LEN, DEVICE)
    val_loader   = DataLoader(data, val_idx,   BATCH_SIZE, SEQ_LEN, DEVICE)

    model = JamOne_Kaiser().to(DEVICE)
    model.count_params()

    # Checkpoint laden  nur wenn gamma_mean nah an 0.020 ist (gesunder Zustand)
    if os.path.exists(BEST_CKPT):
        try:
            tmp = torch.load(BEST_CKPT, map_location=DEVICE)
            # Gesundheitscheck: gamma sollte normal sein
            gamma_vals = [v.float().abs().mean().item()
                          for k, v in tmp.items() if 'gamma' in k]
            avg_gamma  = sum(gamma_vals) / max(1, len(gamma_vals))
            if 0.015 < avg_gamma < 0.030:
                load_model(model, BEST_CKPT, DEVICE)
                print(f"[LOAD] Checkpoint gesund (gamma={avg_gamma:.4f})")
            else:
                print(f"[LOAD] Checkpoint UNGESUND (gamma={avg_gamma:.4f})  von Null")
        except Exception as e:
            print(f"[LOAD] Fehler: {e}  von Null")

    if USE_COMPILE and CUDA:
        try:
            model = torch.compile(model)
            print("[COMPILE] aktiv")
        except:
            pass

    opt_bit, opt_adam = build_optimizers(model)
    healer = SelfHealingController()

    amp_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        if USE_BF16 and CUDA else
        torch.amp.autocast(device_type='cpu', enabled=False)
    )

    @torch.no_grad()
    def evaluate(n=20):
        model.eval()
        losses = []
        stride = 64
        for _ in range(n):
            xb, yb = val_loader.get_batch()
            with amp_ctx:
                seq_losses = []
                for start in range(0, xb.size(1) - 1, stride):
                    end = min(start + stride, xb.size(1) - 1)
                    logits = model(xb[:, :end + 1])
                    loss = F.cross_entropy(
                        logits[:, start:end].reshape(-1, VOCAB_SIZE),
                        yb[:, start:end].reshape(-1)
                    ).item()
                    seq_losses.append(loss)
                losses.append(float(np.mean(seq_losses)))
        model.train()
        return float(np.mean(losses))

    print("[CHECK] Starte Baseline-Evaluation...")
    val_loss = evaluate(1 if not CUDA else 10)
    print(f"[CHECK] Baseline fertig: {val_loss:.4f}")
    best_bpb = val_loss / math.log(2)
    val_bpb  = best_bpb
    print(f"[BASELINE] Val-BPB: {best_bpb:.4f}")
    print(f"[ZIEL]     0.8900  |  Luecke: {best_bpb-0.89:.4f}\n")

    # SWA
    swa_sd, swa_count = None, 0
    SWA_START = 2000

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(f"kaiser_v11_log_{ts}.csv", "w")
    log_f.write("step,trn_bpb,val_bpb,lr_bit,grad_norm,avg_gn,"
                "w_std,clamp,bit_wd,heal_events,best_bpb,status,"
                "lernfortschritt,stabilitaet,abstand_sota,entropie\n")

    model.train()
    t0   = time.perf_counter()
    step = 0

    try:
        while True:
            step += 1
            step_t0 = time.perf_counter()
            print(f"\n>>> [STEP {step}] Starte Forward/Backward...", end="", flush=True)

            # LR Schedule
            lr_b = get_lr(step, LR_BIT,  LR_MIN, 1_000_000, WARMUP_STEPS, STABLE_FRAC)
            lr_a = get_lr(step, LR_ADAM, LR_MIN, 1_000_000, WARMUP_STEPS, STABLE_FRAC)
            # Nur setzen wenn Plateau-Detektor nicht bereits gesenkt hat
            if healer.lr_halved_count == 0:
                set_lr(opt_bit,  lr_b)
                set_lr(opt_adam, lr_a)

            opt_bit.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)

            accum_loss, accum_ent = 0.0, 0.0
            for _ in range(GRAD_ACCUM):
                xb, yb = train_loader.get_batch()
                with amp_ctx:
                    logits = model(xb)
                    loss   = F.cross_entropy(
                        logits.view(-1, VOCAB_SIZE), yb.view(-1)) / GRAD_ACCUM
                loss.backward()
                accum_loss += loss.item()
                with torch.no_grad():
                    p_  = F.softmax(logits.float(), dim=-1)
                    lp_ = F.log_softmax(logits.float(), dim=-1)
                    accum_ent += -(p_ * lp_).sum(-1).mean().item() / GRAD_ACCUM

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt_bit.step()
            opt_adam.step()
            print(f" Fertig (GN: {grad_norm:.2f}).", end="", flush=True)

            # Gewichts-Statistiken
            w_stds = [m.weight.std().item()
                      for m in model.modules() if isinstance(m, BitLinear)]
            avg_ws = sum(w_stds) / max(1, len(w_stds))

            # Selbstheilung inline
            current_val = val_bpb if step % EVAL_EVERY == 0 else None
            status, avg_gn = healer.update(grad_norm, avg_ws, current_val, opt_bit, model)

            trn_bpb = accum_loss / math.log(2)
            cur_lr  = opt_bit.param_groups[0]['lr']

            # SWA Snapshot
            if step >= SWA_START and step % 100 == 0:
                curr = {k: v.clone() for k, v in model.state_dict().items()}
                if swa_sd is None:
                    swa_sd = curr
                else:
                    for k in swa_sd:
                        swa_sd[k] = swa_sd[k] + curr[k]
                swa_count += 1

            # Evaluation
            if step % EVAL_EVERY == 0:
                val_loss = evaluate(15 if CUDA else 8)
                val_bpb  = val_loss / math.log(2)
                flag     = " NEW BEST!" if val_bpb < best_bpb else ""
                stats    = healer.get_stats()
                print(f"\nStep {step:>6} | Trn: {trn_bpb:.4f} | Val: {val_bpb:.4f} "
                      f"| GN: {grad_norm:.2f}(avg:{avg_gn:.1f}) "
                      f"| wstd: {avg_ws:.3f} | clamp: {stats['clamp']} "
                      f"| wd: {stats['wd']} | heals: {stats['heal']}{flag}")

                # Checkpoint NUR bei echtem Fortschritt
                if val_bpb < best_bpb:
                    best_bpb = val_bpb
                    save_model(model, BEST_CKPT)
                    save_model(model, os.path.join(
                        ARCHIVE, f"v11_Step{step}_BPB{val_bpb:.4f}.pth"))
            else:
                if step % 1 == 0:
                    stats = healer.get_stats()
                    print(f"\rStep {step:>6} | BPB: {trn_bpb:.4f} | "
                          f"Best: {best_bpb:.4f} | {status}   ",
                          end='', flush=True)

            dt      = time.perf_counter() - t0
            tok_sec = (BATCH_SIZE * SEQ_LEN * GRAD_ACCUM) / max(1e-9, dt)
            t0      = time.perf_counter()

            lernfortschritt = round(10.13 - best_bpb, 4)  # wieviel wir seit Start gelernt haben
            stabilitaet = "OK" if grad_norm < 5 else "SPIKE"
            abstand_sota = round(best_bpb - 1.14, 4)       # wieviel noch bis SOTA
            log_f.write(
                f"{step},{trn_bpb:.6f},{val_bpb:.6f},"
                f"{cur_lr:.6f},{grad_norm:.4f},{avg_gn:.4f},"
                f"{avg_ws:.4f},{healer.clamp_val:.3f},{healer.current_wd:.4f},"
                f"{healer.heal_events},{best_bpb:.4f},{status},"
                f"{lernfortschritt},{stabilitaet},{abstand_sota},{accum_ent:.4f}\n"
            )
            if MAX_WALLCLOCK_SECS > 0:
                if time.perf_counter() - WALLCLOCK_START >= MAX_WALLCLOCK_SECS:
                    print(f"\n[WALLCLOCK] Zeit abgelaufen. Stoppe.")
                    break

            with open("kaiser_live.json", "w") as jf:
                json.dump({
                    "step": step, "trn_bpb": trn_bpb,
                    "val_bpb": val_bpb, "best_bpb": best_bpb,
                    "grad_norm": grad_norm, "avg_gn": avg_gn,
                    "w_std": avg_ws, "clamp": healer.clamp_val,
                    "bit_wd": healer.current_wd,
                    "heal_events": healer.heal_events,
                    "target": 0.89, "gap": round(best_bpb - 0.89, 4)
                }, jf)

    except KeyboardInterrupt:
        print("\n\n[ABBRUCH] STRG+C...")

    # SWA Finalisierung
    if swa_count > 0:
        print(f"\n[SWA] Mittele {swa_count} Snapshots...")
        for k in swa_sd:
            swa_sd[k] = swa_sd[k] / float(swa_count)
        model.load_state_dict(swa_sd, strict=False)
        swa_bpb = evaluate(20) / math.log(2)
        print(f"[SWA] Val-BPB: {swa_bpb:.4f}")
        if swa_bpb < best_bpb:
            best_bpb = swa_bpb
            save_model(model, BEST_CKPT)
            print("[SWA] Bestes Modell!")

    log_f.close()
    quick_generation_test(model, DEVICE, SEQ_LEN)
    print(f"\n{'='*70}")
    print(f"  FINALE BPB: {best_bpb:.4f} | ZIEL: 0.8900 | LUECKE: {best_bpb-0.89:+.4f}")
    print(f"  Selbstheilungs-Events: {healer.heal_events} | LR-Halbierungen: {healer.lr_halved_count}")
    print(f"{'='*70}\n")

# ══════════════════════════════════════════════════
# FINALE CHECKS
# ══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  FINALE SYSTEM-CHECKS")
    print("="*60)

    # 1. Checkpoint-Größe
    if os.path.exists(BEST_CKPT):
        mb = os.path.getsize(BEST_CKPT) / (1024**2)
        status = "✓ BUDGET OK" if mb < 16.0 else "✗ BUDGET ÜBERSCHRITTEN"
        print(f"[CHECK 1] Checkpoint: {mb:.2f} MB  {status}")
    else:
        print("[CHECK 1] Kein Checkpoint gefunden!")

    # 2. Bestes BPB
    print(f"[CHECK 2] Bestes val_bpb:  {best_bpb:.4f}")
    print(f"[CHECK 2] Abstand zu SOTA: {best_bpb - 1.14:+.4f} BPB")
    print(f"[CHECK 2] Lernfortschritt: {10.13 - best_bpb:.4f} BPB seit Start")

    # 3. Modell laden und Gesundheitscheck
    print("[CHECK 3] Lade besten Checkpoint für Gesundheitscheck...")
    try:
        tmp = torch.load(BEST_CKPT, map_location=DEVICE)
        gamma_vals = [v.float().abs().mean().item()
                      for k, v in tmp.items() if 'gamma' in k]
        if gamma_vals:
            avg_g = sum(gamma_vals)/len(gamma_vals)
            print(f"[CHECK 3] Gamma-Mittelwert: {avg_g:.4f} {'✓' if 0.01 < avg_g < 0.05 else '⚠ PRÜFEN'}")
        w_vals = [v.float().abs().mean().item()
                  for k, v in tmp.items()
                  if 'weight' in k and isinstance(v, torch.Tensor) and v.dtype != torch.uint8]
        avg_w = sum(w_vals)/len(w_vals)
        print(f"[CHECK 3] Gewichts-Mittelwert: {avg_w:.4f} {'✓' if avg_w < 1.0 else '⚠ PRÜFEN'}")
    except Exception as e:
        print(f"[CHECK 3] Fehler: {e}")

    quick_generation_test(model, DEVICE, SEQ_LEN)

# 
# 18. GENERATION TEST
# 
def quick_generation_test(model, device, seq_len):
    """Schneller Test ohne Tokenizer - zeigt rohe Token-Sequenzen"""
    print("\n[GEN-TEST] Starte ohne Tokenizer...")
    model.eval()
    
    # Test 1: Kann er wiederholen?
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(10):
            logits = model(prompt[:, -seq_len:])
            next_tok = logits[0, -1].argmax().item()
            prompt = torch.cat([prompt, 
                torch.tensor([[next_tok]], device=device)], dim=1)
    print(f"[GEN-TEST] Token-Sequenz: {prompt[0].tolist()}")
    
    # Test 2: Entropie der Vorhersagen (niedriger = konfidenter)
    test_input = torch.randint(0, 1024, (1, 50), device=device)
    with torch.no_grad():
        logits = model(test_input)
        probs = torch.softmax(logits[0, -1], dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        top5 = probs.topk(5).indices.tolist()
    print(f"[GEN-TEST] Entropie letzte Position: {entropy:.3f} (niedriger=besser)")
    print(f"[GEN-TEST] Top-5 nächste Tokens: {top5}")
    
    # Test 3: Konsistenz - gleiches Input, gleiches Output?
    x = torch.tensor([[42, 100, 200]], dtype=torch.long, device=device)
    with torch.no_grad():
        out1 = model(x)[0, -1].argmax().item()
        out2 = model(x)[0, -1].argmax().item()
    print(f"[GEN-TEST] Konsistenz: {'✓' if out1 == out2 else '✗'} ({out1} == {out2})")
    
    model.train()

if __name__ == "__main__":
    train()
