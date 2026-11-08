# Roughness (IRI, m/km) from the drive phone's IMU — method research

> **Correction 2026-11-09 (owner):** the drives were recorded on a **scooter** with the phone on the
> handlebar/front, not in a car with a windshield mount. §1–§5 below were written assuming a car; §6 says
> what changes. The code, the features and the stored numbers are unchanged; their interpretation and the
> uncertainty are in §6 and UNCERTAINTY.md.

Written 2026-11-08 for the IRC:82-2023 condition report (PCI weight of roughness = 0.40). Every source
below was opened in this session unless marked "(not opened)"; where a page could not be fetched the claim
is attributed to the page that was read.

## 1. What IRI is, and why a phone cannot measure it directly

- IRI is defined on a **longitudinal profile** (one wheel path): a reference quarter-car (the "Golden Car")
  is driven over the profile at 80 km/h in simulation; IRI = accumulated suspension stroke / distance
  (m/km). The 80 km/h speed is a property of the simulation only; profile-based (IQL-1) systems are
  speed-independent, while **response-type (IQL-3) systems use speed-specific correlation equations**
  (https://en.wikipedia.org/wiki/International_roughness_index).
- Golden Car constants (per unit sprung mass): k_t/m_s = 653, k_s/m_s = 63.3 s^-2, c_s/m_s = 6.0 s^-1,
  m_u/m_s = 0.15, V = 80 km/h. Stated as "k/ms = 63.3, kt/ms = 653, c/ms = 6, mus/ms = 0,15 … little book
  of profiling p.53" in https://uk.mathworks.com/matlabcentral/answers/792577 (secondary); the same car in
  normalised form (2ζ = 0.75, ω_s = 8.0 rad/s, k̄ = 10.3, m̄ = 0.15, V_REF = 80 km/h) in Botshekan et al.
  2020 (link in §2.5). The primary source, Sayers 1995, TRR 1501
  (https://onlinepubs.trb.org/Onlinepubs/trr/1995/1501/1501-001.pdf) returned HTTP 500 when fetched — not
  read. We use 653.2 / 63.3 / 6.0 / 0.15 (ASTM E1926 values as commonly quoted; 653 vs 653.2 changes IRI by
  < 0.1 %), and the 250 mm moving average of the profile.
- A phone in a car is a **response-type system**: it measures the response of *this* car, at *this* speed,
  through *this* mount, not the profile. Any absolute IRI from it is a calibration statement.

## 2. Established methods

### 2.1 Frequency-band magnitudes + speed, linear regression (Douangphachanh & Oneyama)
- "Exploring the Use of Smartphone Accelerometer and Gyroscope to Study on the Estimation of Road Surface
  Roughness Condition", ICINCO 2014 (https://www.scitepress.org/Papers/2014/51174/51174.pdf, text read).
  4 phones (dashboard glued, shirt pocket, box near gearshift) in 4 vehicles (2 pick-ups, 2 sedans) in Laos,
  100 Hz; FFT magnitudes per 100 m section, high-pass filtered; reference IRI from VIMS (itself an
  accelerometer-based system, 10 m IRI, fails below 20 km/h); sections with stops or lengths outside 90–110 m
  dropped. IRI ≈ linear in (band magnitude, average speed); best band reported 40–50 Hz (i.e. just under
  Nyquist of 100 Hz sampling). Adding gyroscope magnitudes raised R² (e.g. 0.575 → 0.766); "almost all"
  R² with gyro > 0.74, best 0.80–0.82. **One regression per phone × vehicle**.
- Needs: reference IRI on calibration sections for every vehicle/phone/mount; speeds above ~20 km/h.

### 2.2 RoadLab Pro (World Bank) / Roadroid (eIRI, cIRI) — commercial response-type apps
- RoadLab Pro on the Hanoi–Hai Phong expressway vs a laser profilometer: Hoang & Nguyen 2019, Transport
  and Communications Science Journal 70(4):352–363 (https://tcsj.utc.edu.vn/index.php/tcgtvt/article/download/485/406,
  text read). On the same lane, mean IRI by the same phone rose **2.98 → 4.77 m/km (+60 %) when speed went
  from 20 to 50 km/h**; the spread between three mounting positions grew with speed (SD 0.18 / 0.27 / 0.46 m/km
  at 20 / 35 / 50 km/h). Conclusion: vehicle speed has an important effect.
- Roadroid: eIRI from peak/RMS of vibration, cIRI from a quarter-car simulation; a correlation > 0.85 with a
  laser system (Forslöf & Jones 2015) and the IQL-3/4 accuracy band of ±20 % are quoted in secondary
  sources found by search (https://centaur.reading.ac.uk/118510/1/B1005031419.pdf — fetch blocked by a bot
  check; not read). Vendor claims of "92 %" (SoftTeco, RoadLab) have no published data behind them.
- Needs: vehicle category / calibration, speed window (RoadLab: 15–100 km/h, per search snippets, not read).

### 2.3 Accumulated vertical displacement / distance (Zang et al. 2018)
- "Assessing and Mapping of Road Surface Roughness based on GPS and Accelerometer Sensors on Bicycle-Mounted
  Smartphones", Sensors 18(3):914 (https://pmc.ncbi.nlm.nih.gov/articles/PMC5876687/, read).
  Vertical = A·Ā/|Ā| with Ā the gravity vector measured while stationary; 0.5 Hz high-pass; distance from
  GPS **speed** (1 Hz) rather than positions (speed accuracy 0.1–0.2 m/s vs ~10 m position);
  IRI ≈ ∬|a_v| dt² / S. Bicycle treated as rigid (no suspension). Validation vs a car-mounted laser
  profiler on 20 m units: r = 0.893 (n = 9); repeated groups differed by 1.80 m/km on average.
- Not transferable as an absolute: a car has a suspension, which is exactly what IRI is about.

### 2.4 Double-integration to a profile, then ProVAL (Islam et al. 2014)
- Islam, Buttlar, Aldunate, Vavrik, "Measurement of Pavement Roughness Using Android-Based Smartphone
  Application", TRR 2457:30–38 (abstract read at
  https://experts.illinois.edu/en/publications/measurement-of-pavement-roughness-using-android-based-smartphone-/).
  Vertical acceleration double-integrated to a "profile", IRI computed in ProVAL; good agreement with an
  inertial profiler on 3 sites and good repeatability, but the authors note rougher pavements **need
  calibration because suspension damping is not accounted for**. Follow-up (Stribling et al., TRB 2017,
  https://trid.trb.org/view/1439271, abstract read): "clear sensitivity" to vehicle size and speed.

### 2.5 Random-vibration inversion with a vehicle model (Botshekan et al. 2020)
- "Roughness-induced vehicle energy dissipation from crowdsourced smartphone measurements through random
  vibration theory", Data-Centric Engineering 1:e16, doi 10.1017/dce.2020.17
  (https://www.cambridge.org/core/journals/data-centric-engineering/article/roughnessinduced-vehicle-energy-dissipation-from-crowdsourced-smartphone-measurements-through-random-vibration-theory/C805260FC79E2D75F267B24FAF0731F9, read;
  record https://dspace.mit.edu/handle/1721.1/133103).
  Road = power-law PSD S(Ω) = g(Ω/Ω₀)^-w; phone acceleration PSD = ω⁴|H_z(ω)|² S(ω) with ω = ΩV;
  **IRI = χ·sqrt(∫ Ω²|H_GC|² S(Ω) dΩ), χ = sqrt(2/π)** for a Gaussian profile; the half-car's 13 parameters
  and the phone position are *fitted* (L2-regularised) from many trips. Reported absolute relative error
  peaks near 10 % against laser IRI (calibrated on one track, validated on another); phone position only
  marginally matters.
- Needs: many trips per vehicle to identify the vehicle, or a calibration track.

### 2.6 Road Impact Factor (Bridgelall)
- RIF / TWIT: an RMS-type vertical-response index per distance, proportional to IRI with vehicle-specific
  factors; combining speed bands makes it usable at any speed (search snippets of NDSU repository records,
  https://library.ndsu.edu/ir/items/8edc3553-007b-471e-844e-254902aa1358; full text behind a bot check, not
  read). A 2021 study found smartphone-to-smartphone inconsistencies on the same vehicle and speed
  (https://library.ndsu.edu/ir/bitstream/handle/10365/32081/, not read).

## 3. What every method needs, and what we have

| Need | Literature | Our drives |
|---|---|---|
| Vertical acceleration | gravity projection (Zang), fixed mount | `acc` 200 Hz (incl. gravity, device axes) + Android `gravity` 50 Hz; windshield suction mount |
| Distance / speed | GPS speed integration | GNSS Doppler 1 Hz + the video pipeline's own track distance s (anchors in each segment record) |
| Speed range | ≥ 15–20 km/h (RoadLab, VIMS) | mostly 15–30 km/h urban; stops in traffic |
| Reference IRI | profiler / RTRRMS sections per vehicle | **none** — to come from the survey's calibration half |
| Vehicle identity | per-vehicle regression | one car per survey, but `survey.vehicle` is empty in the records |

Observed on 2026-10-27 data: the S21 accelerometer magnitude has median 10.9 m/s² while moving (gravity is
9.81) — a scale/bias error of the order of 10 % that a high-pass filter does not remove from the dynamic
part. Another reason any absolute number needs calibration.

## 4. Choice

**Feature (primary):** band-limited RMS of the vertical acceleration at the phone, a_v = acc · ĝ (ĝ =
Android gravity vector), resampled to a uniform 200 Hz grid, zero-phase Butterworth band-pass 0.5–20 Hz
(0.5 Hz removes speed changes and pitch on braking, as Zang; 15 Hz keeps the engine out — see below — and
stays well below Nyquist, unlike 40–50 Hz at 100 Hz sampling), computed over the time the vehicle spends on
each 5 m segment (from the pipeline's track distance), with GNSS speed. Secondary features: body band
0.5–5 Hz, wheel band 5–15 Hz, roll+pitch rate RMS (gyro minus yaw; Douangphachanh found gyro adds
information), peak |a_v|.

**Speed normalisation:** rms / sqrt(v). Reason: for a road whose slope spectrum is white (ISO 8608 w = 2,
the standard assumption also behind the Golden-Car IRI ↔ PSD relation), the RMS of any linear vehicle's
acceleration over a fixed temporal band scales exactly with sqrt(v·G_s) (derivation in app/pci_iri.py).
Whether that removes the speed dependence on *our* car is tested in VALIDATION.md (within-location speed
exponent), not assumed.

**Uncalibrated physical estimate ("Golden-Car-equivalent IRI", `iri_gc_uncal`):** invert the same
random-vibration relation as Botshekan et al. (§2.5) with fixed, stated assumptions instead of a fitted
vehicle: (1) the car's body responds like the Golden Car sprung mass, (2) the phone is rigid on the sprung
mass (no mount resonance, no pitch lever arm), (3) road slope PSD white in the band (w = 2), Gaussian,
(4) one wheel path, (5) the accelerometer's scale is right. Then σ_a² = G_s·v·∫(2πf)²|H_s(f)|²|B(f)|²df and
IRI = 1000·sqrt(2/π)·sqrt(G_s·V·∫|H_s−H_u|²·sinc²(πf·0.25/V)df)/V, so IRI_gc = K(v)·σ_a with K computed
numerically. None of (1)–(5) holds for a hatchback/sedan at 20 km/h with a windshield suction mount; the
number is reported only as a physically-scaled index, labelled uncalibrated, never as IRI.

**Calibrated IRI:** only through `eval_results/pci/iri/iri_calibration.json`, fitted on the survey's
calibration half: log IRI = a + b·log(rms) + c·log(v) (+ optional d·log(rms_rot)) by least squares on
100 m sections (the survey's unit; 5 m IRI is not a recognised quantity), with the fitted range of rms and
speed stored and applied only inside it. Without the file the report says "uncalibrated".

**Why not the alternatives:** double integration (§2.4) is unstable and still needs suspension calibration;
full vehicle identification (§2.5) needs many trips of one car and a reference to check it; a commercial
app cannot be re-run on recorded data. The chosen method is the common core of §2.1–2.6 (band RMS + speed
+ per-vehicle calibration), with the physical scaling made explicit so that the calibration has few
parameters.

**A windshield mount and an unknown car make absolute IRI a calibration problem.** The suction mount adds a
resonance (typically 10–30 Hz) and records pitch × lever arm; the car's suspension, tyres and load set the
gain; the accelerometer's scale is off by ~10 %. All three are constant for one survey car + mount, which is
why a per-survey calibration (§4) can work, and why it must be refitted if the car, mount position or phone
changes.

## 5. Band choice revised from the data (2026-11-08, before any validation result was looked at)

The first draft used 0.5–20 Hz. The Welch spectrum of a_v on drive 20261027T101057Z_…_3d9b (whole drive)
has its largest line at **24.4–25.1 Hz** (engine firing near idle: ~735 rpm × 2 for a 4-cylinder) and more
energy above 20 Hz (RMS 1.27 m/s² in 20–30 Hz, 1.0–1.3 in each band up to 100 Hz) than below it (0.2–0.5
per band in 0.5–20 Hz). A 4th-order zero-phase Butterworth with a 20 Hz corner passes ~14 % of the 25 Hz
amplitude; with a 15 Hz corner ~1.6 %. The main band is therefore **0.5–15 Hz** (wheel band 5–15 Hz). The
Golden Car's wheel-hop mode (~10.5 Hz) is inside it, and the band filter is part of the K(v) integral, so the
uncalibrated index stays consistent. The raw accelerometer has per-axis SD of 3–6 m/s² while driving —
almost all of it above 20 Hz (engine, mount) — which is why an unfiltered RMS (as in eIRI-style apps) would
mostly measure the engine.

## 6. The vehicle is a scooter (owner, 2026-11-09) — what changes

- **Physics.** A scooter is single-track (front and rear wheel in one line, ~1.25 m apart), with a stiffer,
  lighter suspension relative to its sprung mass (frame + rider), small 10–12 in wheels with stiff tyres
  (higher wheel-hop frequency, less tyre enveloping of short bumps), a heavy unsprung engine/swing-arm unit,
  a rider who is a second sprung mass, steering-induced roll, and a handlebar/holder with its own modes. The
  Golden-Car index (`iri_gc_uncal`) therefore does not measure IRI even under ideal conditions;
  `eval_results/pci/iri/sensitivity.py` puts its ratio to the true IRI at 0.66–3.4 (90 % of plausible
  scooters, median ~1.6) — UNCERTAINTY.md §3.
- **Band.** The moving spectra of every drive show a fixed ~24–25 Hz line (handlebar/holder/frame mode — it
  does not follow speed) and engine lines at ~64–70 Hz; at stops ~26–31 and ~54–64 Hz (NOISE_FLOOR.md D1).
  All are outside 0.5–15 Hz, so the band stays. (The §5 reading of 25 Hz as a car engine at idle was wrong.)
- **Speed normalisation.** rms/sqrt(v) follows from a white-slope road for any linear vehicle, so it holds
  for a scooter in the same sense as for a car; the wheelbase filter (front and rear wheel hitting the same
  bump L/v apart) adds speed dependence. The within-location speed exponent is still unmeasured (VALIDATION V4).
- **Literature.** The one direct scooter-vs-car comparison found (Setiawan & Nurdin 2019, Yamaha Mio vs a
  car on the same Bandung route) reads ~1.9× the car's IRI from the scooter and agrees much worse with visual
  ratings; no calibrated motorcycle IRI model with error bands was found (UNCERTAINTY.md §1).
- **Rut depth** cannot come from a single-track vehicle's roll (eval_results/pci/rut/RUT_PRIOR.md).
- Every output now carries `vehicle` (pci_iri.VEHICLE).
