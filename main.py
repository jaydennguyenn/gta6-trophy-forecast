"""GTA VI trophy earn-rate forecaster for make.fun's
gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13, standard library only).

Reads every case file <id>.in in the input directory and writes one forecast
file <id>.out to the output directory: for each of the case's trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target, in percent.

Method, per trophy, on the logit scale (logit(p) = ln(p / (100 - p)) with the
rate clamped to 0.01..99.99 percent):
  * Time is measured in hours since the game's launch (2026-11-19T00:00:00Z)
    and compressed with x = ln(1 + hours), so fast early growth slows down.
  * Reads are weighted by recency (24-hour half-life): the last day of reads
    says most about where a rate is heading.
  * Two trend models are fitted and the better one (by weighted squared error
    on the logit scale) is used; when they agree within 10% their medians are
    averaged in logit space:
      - a straight line of logit(rate) against ln(1 + hours);
      - a saturating curve: rate = A * Phi((ln(1 + hours) - mu) / sigma),
        fitted by gridding the asymptote A and regressing the probit of
        rate / A on ln(1 + hours). The asymptote is the share of players
        who will ever earn the trophy, so the curve levels off instead of
        running away.
  * The spread around the median combines the trend's own prediction error,
    a random-walk term that grows with the horizon (0.5 logit per root-day),
    and a fixed jump term: Exophase refreshes its counts in jumps, and one
    refresh can move a rate's logit by a sizeable amount, so the band is
    never narrower than that jump allows.

The program is deterministic: fixed grids, sorted iteration, no randomness,
no wall-clock reads. It runs in well under the 8-second budget.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The quantile levels each forecast row lists, and the standard normal's
# quantile at each of them (computed below from the case's own levels).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Spread terms, in logit units.
RESID_FLOOR = 0.15      # a trend never fits perfectly; floor on residual std
SIGMA_PER_ROOT_DAY = 0.4  # random-walk widening with the horizon
JUMP_SIGMA = 0.35        # one Exophase refresh can jump a rate's logit
MAX_MOVE = 1.5          # cap on how far the median may move from the last rate
MAX_SIGMA = 2.0         # cap on the total spread
MIN_SIGMA = 0.25        # the band is never narrower than this


def clamp_rate(pct):
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    """Back to a percent, guarded against overflow either way."""
    if y > 20.0:
        return 100.0
    if y < -20.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-y))


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def parse_time(text):
    """ISO 8601 to an aware datetime; accepts a trailing Z."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def weighted_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b, sxx, xbar, wsum)."""
    wsum = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / wsum
    ybar = sum(w * y for y, w in zip(ys, ws)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return ybar, 0.0, sxx, xbar, wsum
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return ybar - b * xbar, b, sxx, xbar, wsum


def fit_log_linear(points, target_hours):
    """Straight line of logit(rate) on ln(1 + hours). Returns (median_logit, sse, resid_std, sxx, xbar, wsum)."""
    xs = [math.log1p(h) for h, _, _ in points]
    ys = [logit(r) for _, r, _ in points]
    ws = [w for _, _, w in points]
    a, b, sxx, xbar, wsum = weighted_line(xs, ys, ws)
    pred = [a + b * x for x in xs]
    sse = sum(w * (y - p) ** 2 for y, p, w in zip(ys, pred, ws))
    resid = math.sqrt(max(sse / wsum, 0.0))
    return a + b * math.log1p(target_hours), sse, resid, sxx, xbar, wsum


def fit_saturating(points, x_target):
    """rate = A * Phi((ln(1+h) - mu) / sigma), A gridded; returns the best
    (median_logit, sse, resid_std, sxx, xbar, wsum) on the logit scale, or None
    when the history cannot support it.

    The asymptote is only weakly identified by a still-rising history: a huge
    A can fit the observed window as well as the true one while extrapolating
    wildly. So among the asymptotes whose fit is within 10% of the best, the
    SMALLEST is used for the forecast (the most conservative reading of the
    data), while the best fit's error decides the model comparison."""
    rates = [r for _, r, _ in points]
    top = max(rates)
    if top <= 0.0:
        return None
    xs = [math.log1p(h) for h, _, _ in points]
    ws = [w for _, _, w in points]
    fits = []
    factors = [1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    for f in factors:
        asym = top * f
        if asym >= 100.0:
            asym = 100.0
        zs = []
        ok = True
        for _, r, _ in points:
            # A printed 0.00% means "below half a hundredth": treat it as the
            # smallest representable rate rather than dropping the asymptote.
            p = max(r, 0.01) / asym
            if p >= 1.0:
                ok = False
                break
            zs.append(norm_ppf(p))
        if not ok:
            continue
        a, b, sxx, xbar, wsum = weighted_line(xs, zs, ws)
        z_target = a + b * math.log1p(x_target)
        med_rate = asym * norm_cdf(z_target)
        # Score this asymptote by the logit-scale squared error of its fit.
        sse = 0.0
        for (h, r, w) in points:
            pred_rate = asym * norm_cdf(a + b * math.log1p(h))
            sse += w * (logit(r) - logit(pred_rate)) ** 2
        resid = math.sqrt(max(sse / wsum, 0.0))
        fits.append((asym, logit(med_rate), sse, resid, sxx, xbar, wsum))
        if asym >= 100.0:
            break
    if not fits:
        return None
    best_sse = min(fit[2] for fit in fits)
    for fit in fits:  # fits are in increasing asymptote order
        if fit[2] <= 1.1 * best_sse:
            _, med, _sse, resid, sxx, xbar, wsum = fit
            return (med, best_sse, resid, sxx, xbar, wsum)
    return None


def forecast_trophy(rates, read_hours, target_hours, horizon_days):
    """rates: list of (rate or None) per read; read_hours: hours since launch
    per read; target_hours: hours since launch at the target. Returns the five
    quantiles in percent."""
    last = None
    for r in reversed(rates):
        if r is not None:
            last = r
            break
    if last is None:
        last = 1.0
    points = []
    h_last = read_hours[-1] if read_hours else 0.0
    for h, r in zip(read_hours, rates):
        if r is None:
            continue
        w = 0.5 ** ((h_last - h) / 24.0)
        points.append((h, r, w))
    last_logit = logit(last)
    if len(points) < 3 or len({round(math.log1p(h), 6) for h, _, _ in points}) < 3:
        # Too little history to fit a trend: persist the last rate.
        sigma = max(MIN_SIGMA, math.sqrt(0.25 * horizon_days + JUMP_SIGMA ** 2 + RESID_FLOOR ** 2))
        return quantiles_from(last_logit, sigma)
    x_target = math.log1p(target_hours)
    med_a, sse_a, resid_a, sxx_a, xbar_a, wsum_a = fit_log_linear(
        [(h, r, w) for h, r, w in points], target_hours)
    sat = fit_saturating(points, target_hours)
    if sat is not None and sat[1] < 1.1 * sse_a:
        # The saturating curve is the physically correct family (earn rates
        # plateau as the player base is exhausted) and it fits at least as
        # well as the unbounded log-linear trend: trust it outright. Blending
        # in the log-linear median was the failure mode on sparse histories,
        # where the log-linear fit extrapolates the early steep rise to ~96%
        # for a trophy that is already flat at ~56%.
        med, sse, resid, sxx, xbar, wsum = sat
    else:
        med, sse, resid, sxx, xbar, wsum = med_a, sse_a, resid_a, sxx_a, xbar_a, wsum_a
    # Never let the median run away from the last observed rate.
    med = min(max(med, last_logit - MAX_MOVE), last_logit + MAX_MOVE)
    resid = max(resid, RESID_FLOOR)
    # Prediction error of the trend, growing with the extrapolation distance.
    if sxx > 1e-12:
        leverage = (math.log1p(target_hours) - xbar) ** 2 / sxx
    else:
        leverage = 0.0
    se = resid * math.sqrt(1.0 + 1.0 / max(wsum, 1.0) + leverage)
    sigma = math.sqrt(se * se + (SIGMA_PER_ROOT_DAY ** 2) * horizon_days + JUMP_SIGMA ** 2)
    sigma = min(max(sigma, MIN_SIGMA), MAX_SIGMA)
    return quantiles_from(med, sigma)


def quantiles_from(median_logit, sigma):
    z_scores = [norm_ppf(level) for level in LEVELS]
    out = []
    prev = None
    for z in z_scores:
        value = inv_logit(median_logit + z * sigma)
        value = round4(value)
        if prev is not None and value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def fallback_forecast(case):
    """Persistence forecast used only if a case cannot be modelled at all."""
    history = case.get("history") or []
    last_rates = history[-1]["rates"] if history else []
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 1.0
        rows.append({"id": trophy["id"], "q": quantiles_from(logit(rate), 1.0)})
    return {"trophies": rows}


def forecast_case(case):
    history = case.get("history") or []
    if not history:
        return fallback_forecast(case)
    target = parse_time(case["target"])
    target_hours = hours_since_launch(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    read_hours = []
    for read in history:
        stamp = read.get("at") or read.get("slot")
        read_hours.append(hours_since_launch(parse_time(stamp)))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rates = []
        for read in history:
            rs = read.get("rates") or []
            rates.append(rs[i] if i < len(rs) else None)
        try:
            qs = forecast_trophy(rates, read_hours, target_hours, horizon_days)
        except Exception:
            qs = quantiles_from(logit(rates[-1] if rates and rates[-1] is not None else 1.0), 1.0)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
