# -*- coding: utf-8 -*-
"""Reproduce notebook analysis without opening GUI plots.
Run in the project directory: python run_analysis.py
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: plt.close('all')
def cli_display(*objects):
    for obj in objects:
        data = getattr(obj, 'data', obj)
        print(data.to_string(index=False) if hasattr(data, 'to_string') else data)

# %%
from pathlib import Path
import hashlib
import json
import platform
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

YEAR = 2020
COUNTRY = "USA"
DISTANCES_KM = {"50km": 50.0, "50mi": 80.4672}  # 1 mile = 1.609344 km
SPEED_TOLERANCE = 0.02  # km/h: kiểm tra sai khác do làm tròn/định dạng
MIN_AGE_OBSERVATIONS = 30  # ngưỡng trình bày mô tả, không phải kiểm định thống kê
DATA_PATH = None  # Ví dụ: Path("TWO_CENTURIES_OF_UM_RACES.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 120,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "font.size": 10})
COLORS = {"F": "#d36582", "M": "#2874a6"}
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

def table(df, decimals=3):
    display(df.style.format(precision=decimals, na_rep="Chưa biết"))

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.show()

print({"Python": platform.python_version(), "pandas": pd.__version__,
       "numpy": np.__version__, "matplotlib": matplotlib.__version__})
display = cli_display

# %%
columns = ["Year of event", "Event dates", "Event name", "Event distance/length",
           "Event number of finishers", "Athlete performance", "Athlete club",
           "Athlete country", "Athlete year of birth", "Athlete gender",
           "Athlete age category", "Athlete average speed", "Athlete ID"]
candidates = [Path("TWO_CENTURIES_OF_UM_RACES.csv"),
              Path("data/um_usa_2020_50k_50mi.csv"),
              Path("um_usa_2020_50k_50mi.csv")]
path = Path(DATA_PATH) if DATA_PATH is not None else next((p for p in candidates if p.is_file()), None)
if path is None:
    raise FileNotFoundError("Đặt CSV nguồn hoặc CSV phạm vi đi kèm cạnh notebook/trong thư mục data.")
parts, loaded_rows = [], 0
for chunk in pd.read_csv(path, dtype="string", usecols=columns, chunksize=250_000):
    loaded_rows += len(chunk)
    event_country = chunk["Event name"].str.extract(r"\(([A-Z]{3})\)\s*$", expand=False)
    in_scope = (pd.to_numeric(chunk["Year of event"], errors="coerce").eq(YEAR)
                & chunk["Event distance/length"].isin(DISTANCES_KM)
                & event_country.eq(COUNTRY))
    parts.append(chunk.loc[in_scope].copy())
raw = pd.concat(parts, ignore_index=True)
assert not raw.empty, "Không có dòng nào thuộc phạm vi đã chọn."

data_sha256 = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
manifest_path = next((p for p in [Path("data/manifest.json"), Path("manifest.json")] if p.is_file()), None)
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
if path.name == "um_usa_2020_50k_50mi.csv" and manifest:
    assert data_sha256 == manifest["scope_sha256"], "CSV phạm vi không khớp manifest."
    assert len(raw) == manifest["scope_rows"]
display(pd.DataFrame({"Chỉ tiêu": ["Dòng CSV thực sự đọc trong lần chạy này", "Dòng thuộc phạm vi",
                                  "Dung lượng CSV đã đọc (MB)", "SHA-256 CSV đã đọc"],
                      "Giá trị": [loaded_rows, len(raw), round(path.stat().st_size / 1e6, 2), data_sha256]}))
display(raw.head(3))

# %%
old_country = raw["Event name"].str.split("(").str.get(1).str.split(")").str.get(0)
missed = raw.loc[~old_country.eq(COUNTRY).fillna(False)]
filter_difference = missed.groupby(["Event name", "Event distance/length"]).size().rename("Lượt bị bỏ sót").reset_index()
table(filter_difference, 0)
print(f"Lọc cũ: {len(raw)-len(missed):,}; lọc đã sửa: {len(raw):,}; chênh lệch: {len(missed):,} lượt.")

# %%
def parse_event_date(value):
    value = str(value).strip()
    # Ví dụ 02.02.2020; 18.-19.01.2020; 29.02.-01.03.2020
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value):
        start = value
    elif re.fullmatch(r"\d{2}\.-\d{2}\.\d{2}\.\d{4}", value):
        start = value[:2] + "." + value[-7:]
    elif re.fullmatch(r"\d{2}\.\d{2}\.-\d{2}\.\d{2}\.\d{4}", value):
        start = value[:6] + value[-4:]
    else:
        return pd.NaT
    return pd.to_datetime(start, format="%d.%m.%Y", errors="coerce")

# Kiểm tra đúng các định dạng ảnh hưởng trực tiếp tới phân nhóm mùa.
assert parse_event_date("18.-19.01.2020") == pd.Timestamp("2020-01-18")
assert parse_event_date("29.02.-01.03.2020") == pd.Timestamp("2020-02-29")

d = raw.rename(columns={"Year of event": "year", "Event dates": "date_raw",
    "Event name": "race_name_raw", "Event distance/length": "distance",
    "Event number of finishers": "reported_finishers", "Athlete performance": "performance_raw",
    "Athlete year of birth": "birth_year_raw", "Athlete gender": "gender",
    "Athlete average speed": "speed_raw", "Athlete ID": "athlete_id"}).copy()
d["race_name"] = d["race_name_raw"].str.replace(r"\s*\([A-Z]{3}\)\s*$", "", regex=True).str.strip()
d["race_date"] = d["date_raw"].map(parse_event_date)
d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
birth = pd.to_numeric(d["birth_year_raw"], errors="coerce")
age = d["year"] - birth
invalid_age = birth.notna() & (birth.mod(1).ne(0) | age.lt(0) | age.gt(100))
d["age"] = age.mask(invalid_age).astype("Int64")
d["speed_kmh"] = pd.to_numeric(d["speed_raw"], errors="coerce")
d["duration"] = pd.to_timedelta(d["performance_raw"].str.replace(r"\s*h\s*$", "", regex=True), errors="coerce")
d["duration_hours"] = d["duration"].dt.total_seconds() / 3600
d["distance_km"] = d["distance"].map(DISTANCES_KM)
d["speed_calculated"] = d["distance_km"] / d["duration_hours"]
d["speed_difference"] = (d["speed_kmh"] - d["speed_calculated"]).abs()
d["reported_finishers"] = pd.to_numeric(d["reported_finishers"], errors="coerce").astype("Int64")
d["gender"] = d["gender"].str.strip().str.upper()
d["month"] = d["race_date"].dt.month.astype("Int64")
month_to_season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
d["season"] = pd.Categorical(d["month"].map(month_to_season), categories=SEASONS, ordered=True)
d["race_id"] = d["race_name_raw"] + " | " + d["date_raw"] + " | " + d["distance"]
key = ["race_name_raw", "date_raw", "distance", "athlete_id"]
bad_speed = (d["speed_kmh"].isna() | ~np.isfinite(d["speed_kmh"].astype(float))
             | d["speed_kmh"].le(0) | d["duration_hours"].isna()
             | d["duration_hours"].le(0) | d["speed_difference"].gt(SPEED_TOLERANCE)).fillna(True)
quality = pd.DataFrame({"Kiểm tra": ["Tuổi chưa biết", "Tuổi/năm sinh cần xem xét", "Ngày không đọc được",
    "Giới tính ngoài M/F", "Thiếu Athlete ID", "Tốc độ/thời gian cần xem xét",
    "Dòng trùng hoàn toàn", "Khóa kết quả trùng (đếm mọi dòng liên quan)", "Tên giải chứa ký tự thay thế �"],
    "Số dòng": [d["age"].isna().sum(), invalid_age.sum(), d["race_date"].isna().sum(),
    (~d["gender"].isin(["M", "F"])).sum(), d["athlete_id"].isna().sum(), bad_speed.sum(),
    raw.duplicated().sum(), d.duplicated(key, keep=False).sum(), d["race_name_raw"].str.contains("�", regex=False).sum()]})
table(quality, 0)
# Kiểm tra nhận dạng: cùng ID nhưng nhiều năm sinh/giới tính hoặc nhiều kết quả trong cùng phiên.
identity_audit = d.assign(birth_year_numeric=birth).groupby("athlete_id").agg(
    birth_values=("birth_year_numeric", "nunique"), gender_values=("gender", "nunique"))
duplicate_key_ids = set(d.loc[d.duplicated(key, keep=False), "athlete_id"])
ambiguous_ids = set(identity_audit.index[(identity_audit.birth_values > 1) | (identity_audit.gender_values > 1)]) | duplicate_key_ids
identity_conflicts = d.loc[d.athlete_id.isin(ambiguous_ids)].copy()
display(pd.DataFrame({"Kiểm tra nhận dạng": ["ID có nhiều năm sinh đã biết", "ID có nhiều giới tính",
    "ID liên quan tổ hợp kết quả trùng", "ID cần loại khỏi phân tích theo người", "Lượt liên quan các ID này"],
    "Số lượng": [(identity_audit.birth_values > 1).sum(), (identity_audit.gender_values > 1).sum(),
                 len(duplicate_key_ids), len(ambiguous_ids), len(identity_conflicts)]}))
assert not raw.duplicated().any(), "Có dòng trùng hoàn toàn; cần quyết định xử lý trước khi tổng hợp."
assert d["race_date"].notna().all() and d["athlete_id"].notna().all()
assert d["race_date"].dt.year.eq(YEAR).all()
clean = d.loc[~bad_speed].copy().reset_index(drop=True)
quarantine = d.loc[bad_speed].copy()
identity_clean = clean.loc[~clean.athlete_id.isin(ambiguous_ids)].copy()
print(f"Giữ {len(clean):,}/{len(raw):,} lượt cho phân tích tốc độ; cách ly {len(quarantine):,} lượt.")
print(f"Sai lệch tốc độ lớn nhất: {d['speed_difference'].max():.6f} km/h.")

# %%
overview = pd.DataFrame({"Chỉ tiêu": ["Lượt kết quả dùng phân tích tốc độ", "ID nguồn duy nhất (chưa xác minh số người)",
    "Phiên giải/cự ly duy nhất", "Lượt có tuổi", "Lượt thiếu tuổi"],
    "Giá trị": [len(clean), clean.athlete_id.nunique(), clean.race_id.nunique(),
                clean.age.notna().sum(), clean.age.isna().sum()]})
table(overview, 0)
event_counts = clean.groupby("race_id").agg(records=("athlete_id", "size"),
    reported_finishers=("reported_finishers", "first"),
    different_reported_values=("reported_finishers", "nunique"))
print("Phiên có nhiều giá trị số người hoàn thành:", (event_counts.different_reported_values > 1).sum())
print("Phiên có số dòng khác số người hoàn thành nguồn:",
      (event_counts.records != event_counts.reported_finishers).sum())

# %%
participation = clean.groupby(["distance", "gender"]).agg(
    observations=("athlete_id", "size"), source_ids=("athlete_id", "nunique"),
    race_editions=("race_id", "nunique")).reset_index()
participation["share_within_distance_pct"] = 100 * participation.observations / participation.groupby("distance").observations.transform("sum")
table(participation, 2)
counts = participation.pivot(index="distance", columns="gender", values="observations").fillna(0).reindex(["50km", "50mi"])
fig, ax = plt.subplots(figsize=(9, 3.4))
left = np.zeros(len(counts))
for gender in ["F", "M"]:
    values = counts[gender].to_numpy()
    ax.barh(counts.index, values, left=left, color=COLORS[gender], label=gender)
    for i, value in enumerate(values):
        ax.text(left[i] + value / 2, i, f"{int(value):,}", ha="center", va="center", color="white", fontweight="bold")
    left += values
ax.set(xlabel="Lượt kết quả", ylabel="Cự ly", title="Cơ cấu lượt kết quả theo cự ly và giới tính — Mỹ, 2020")
ax.legend(title="Giới tính", loc="lower right")
save_figure(fig, "01_participation.png")

# %%
speed_summary = clean.groupby(["distance", "gender"]).agg(
    observations=("speed_kmh", "size"), source_ids=("athlete_id", "nunique"),
    mean_kmh=("speed_kmh", "mean"), median_kmh=("speed_kmh", "median"),
    q25_kmh=("speed_kmh", lambda s: s.quantile(.25)), q75_kmh=("speed_kmh", lambda s: s.quantile(.75))).reset_index()
table(speed_summary)
groups = [(distance, gender) for distance in ["50km", "50mi"] for gender in ["F", "M"]]
fig, ax = plt.subplots(figsize=(9, 4))
boxes = ax.boxplot([clean.loc[clean.distance.eq(x) & clean.gender.eq(g), "speed_kmh"].astype(float)
                    for x, g in groups], tick_labels=[f"{x} · {g}" for x,g in groups],
                   showfliers=False, patch_artist=True)
for patch, (_, gender) in zip(boxes["boxes"], groups):
    patch.set_facecolor(COLORS[gender]); patch.set_alpha(.65)
ax.set(ylabel="Tốc độ (km/h)", title="Phân bố tốc độ theo cự ly và giới tính")
save_figure(fig, "02_speed_distribution.png")

# %%
AGE_LABELS = ["0–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70–100"]
age_data = clean.loc[clean.age.notna()].copy()
age_data["age_group"] = pd.cut(age_data.age.astype(float), bins=[-1,19,29,39,49,59,69,100], labels=AGE_LABELS)
age_summary = age_data.groupby(["distance", "age_group"], observed=True).agg(
    observations=("speed_kmh", "size"), source_ids=("athlete_id", "nunique"),
    mean_kmh=("speed_kmh", "mean"), median_kmh=("speed_kmh", "median")).reset_index()
age_summary["shown_on_chart"] = age_summary.observations.ge(MIN_AGE_OBSERVATIONS)
table(age_summary)
fig, ax = plt.subplots(figsize=(9, 4))
for distance, color in [("50km", "#2874a6"), ("50mi", "#ca853b")]:
    group = age_summary.loc[age_summary.distance.eq(distance)].set_index("age_group").reindex(AGE_LABELS)
    values = group.median_kmh.where(group.observations.ge(MIN_AGE_OBSERVATIONS))
    ax.plot(AGE_LABELS, values, marker="o", label=distance, color=color)
ax.set(xlabel="Tuổi xấp xỉ", ylabel="Trung vị tốc độ (km/h)", title="Tốc độ theo nhóm tuổi — nhóm có ít nhất 30 lượt")
ax.legend(title="Cự ly")
save_figure(fig, "03_age_groups.png")

# %%
age_missing_sensitivity = clean.assign(age_status=np.where(clean.age.isna(), "Tuổi chưa biết", "Có tuổi")).groupby(
    ["distance", "age_status"]).agg(observations=("speed_kmh", "size"), mean_kmh=("speed_kmh", "mean"),
                                  median_kmh=("speed_kmh", "median")).reset_index()
table(age_missing_sensitivity)

# %%
season_summary = clean.groupby(["distance", "season"], observed=True).agg(
    observations=("speed_kmh", "size"), source_ids=("athlete_id", "nunique"),
    race_editions=("race_id", "nunique"), mean_kmh=("speed_kmh", "mean"),
    median_kmh=("speed_kmh", "median")).reset_index()
event_means = clean.groupby(["distance", "season", "race_id"], observed=True).agg(
    observations=("speed_kmh", "size"), event_mean_kmh=("speed_kmh", "mean")).reset_index()
equal_event = event_means.groupby(["distance", "season"], observed=True).event_mean_kmh.mean().rename("equal_event_mean_kmh").reset_index()
season_summary = season_summary.merge(equal_event, on=["distance", "season"], validate="one_to_one")
table(season_summary)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, distance in zip(axes, ["50km", "50mi"]):
    s = season_summary.loc[season_summary.distance.eq(distance)].set_index("season").reindex(SEASONS)
    x = np.arange(4)
    ax.bar(x-.18, s.mean_kmh, .36, label="Mean theo lượt", color="#2874a6")
    ax.bar(x+.18, s.equal_event_mean_kmh, .36, label="Mỗi phiên giải trọng số bằng nhau", color="#ca853b")
    ax.set(xticks=x, xticklabels=SEASONS, title=distance, xlabel="Mùa theo tháng bắt đầu giải")
    ax.set_ylim(0, max(season_summary.mean_kmh.max(), season_summary.equal_event_mean_kmh.max()) * 1.18)
axes[0].set_ylabel("Tốc độ trung bình (km/h)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.07), ncol=2)
fig.suptitle("So sánh mùa với hai cách đặt trọng số", y=1.03)
save_figure(fig, "04_season_sensitivity.png")

# %%
athlete_stats = identity_clean.groupby(["distance", "gender", "athlete_id"]).agg(
    observations=("speed_kmh", "size"), athlete_mean_kmh=("speed_kmh", "mean")).reset_index()
person_weighted = athlete_stats.groupby(["distance", "gender"]).agg(
    eligible_source_ids=("athlete_id", "size"), repeat_source_ids=("observations", lambda s: s.gt(1).sum()),
    mean_per_athlete_kmh=("athlete_mean_kmh", "mean")).reset_index()
matching_subset = identity_clean.groupby(["distance", "gender"]).speed_kmh.mean().rename("mean_matching_subset_kmh").reset_index()
weight_sensitivity = speed_summary[["distance", "gender", "mean_kmh"]].merge(
    person_weighted, on=["distance", "gender"], validate="one_to_one")
weight_sensitivity = weight_sensitivity.merge(matching_subset, on=["distance", "gender"], validate="one_to_one")
table(weight_sensitivity)

within_race = clean.loc[clean.gender.isin(["M", "F"])].groupby(["distance", "race_id", "gender"]).agg(
    n=("speed_kmh", "size"), mean=("speed_kmh", "mean")).unstack("gender")
eligible = within_race.loc[(within_race[("n", "M")] >= 10) & (within_race[("n", "F")] >= 10)].copy()
eligible["gap_m_minus_f"] = eligible[("mean", "M")] - eligible[("mean", "F")]
within_race_summary = eligible.groupby(level="distance")["gap_m_minus_f"].agg(
    eligible_race_editions="size", mean_gap_kmh="mean", median_gap_kmh="median").reset_index()
table(within_race_summary)

# %%
summary_lines = [f"**Phạm vi:** {len(clean):,} lượt kết quả, {clean.athlete_id.nunique():,} ID nguồn "
                 f"tại {clean.race_id.nunique():,} phiên giải/cự ly tại Mỹ năm {YEAR}.",
    f"**Kiểm tra dữ liệu:** phục hồi {len(missed):,} lượt mà cách lấy ngoặc đầu tiên bỏ sót; "
    f"giữ {clean.age.isna().sum():,} lượt thiếu tuổi cho các phân tích không dùng tuổi. "
    f"{len(quarantine):,} lượt bị cách ly vì kiểm tra tốc độ/thời gian.",
    f"**Nhận dạng:** {len(ambiguous_ids):,} ID mâu thuẫn, liên quan {len(identity_conflicts):,} lượt; "
    "giữ các lượt cho mô tả thành tích, loại những ID này khỏi kiểm tra trọng số theo người. Số ID nguồn không xác nhận số người duy nhất."]
for distance in ["50km", "50mi"]:
    group = speed_summary.loc[speed_summary.distance.eq(distance)].set_index("gender")
    m, f = group.loc["M"], group.loc["F"]
    summary_lines.append(f"**{distance}, giới tính:** trung bình M {m.mean_kmh:.2f} và F {f.mean_kmh:.2f} km/h "
        f"(chênh {m.mean_kmh-f.mean_kmh:.2f}); trung vị M {m.median_kmh:.2f} và F {f.median_kmh:.2f}. "
        "")
    s = season_summary.loc[season_summary.distance.eq(distance)].set_index("season")
    summary_lines.append(f"**{distance}, mùa:** trung bình theo lượt vào mùa hè {s.loc['Summer','mean_kmh']:.2f}, "
        f"mùa đông {s.loc['Winter','mean_kmh']:.2f} km/h; khi cho mỗi phiên giải trọng số bằng nhau, "
        f"hai giá trị là {s.loc['Summer','equal_event_mean_kmh']:.2f} và {s.loc['Winter','equal_event_mean_kmh']:.2f}. "
        "")
    a = age_summary.loc[age_summary.distance.eq(distance) & age_summary.shown_on_chart].sort_values("median_kmh", ascending=False).iloc[0]
    summary_lines.append(f"**{distance}, tuổi:** nhóm {a.age_group} có trung vị cao nhất trong các nhóm đủ ngưỡng "
        f"({a.median_kmh:.2f} km/h, {a.observations:,} lượt). ")
    w = weight_sensitivity.loc[weight_sensitivity.distance.eq(distance)]
    change = (w.mean_per_athlete_kmh - w.mean_matching_subset_kmh).abs().max()
    summary_lines.append(f"**{distance}, trọng số theo ID:** trên mẫu đã loại ID mâu thuẫn, chuyển từ trọng số theo lượt "
        f"sang theo ID làm tốc độ trung bình mỗi nhóm giới tính thay đổi tối đa {change:.3f} km/h. "
        "")
    within = within_race_summary.loc[within_race_summary.distance.eq(distance)].iloc[0]
    summary_lines.append(f"**{distance}, trong cùng phiên giải:** trung vị chênh lệch tốc độ trung bình M−F là "
        f"{within.median_gap_kmh:.2f} km/h, trên {int(within.eligible_race_editions)} phiên đủ ít nhất 10 lượt mỗi nhóm. "
        "")
display(Markdown("\n\n".join(summary_lines)))

# %%
assert len(clean) + len(quarantine) == len(raw)
assert participation.observations.sum() == len(clean)
assert speed_summary.observations.sum() == len(clean)
assert season_summary.observations.sum() == len(clean)
assert age_summary.observations.sum() == clean.age.notna().sum()
assert clean.speed_difference.le(SPEED_TOLERANCE).all()
tables = {"quality_checks": quality, "overview": overview, "filter_difference": filter_difference,
    "participation": participation, "speed_summary": speed_summary, "age_summary": age_summary,
    "age_missing_sensitivity": age_missing_sensitivity, "season_summary": season_summary,
    "weight_sensitivity": weight_sensitivity, "within_race_summary": within_race_summary}
for name, frame in tables.items():
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
clean.drop(columns=["duration"]).to_csv(OUT / "analysis_data.csv", index=False, encoding="utf-8-sig")
identity_conflicts.drop(columns=["duration"]).to_csv(OUT / "identity_conflicts.csv", index=False, encoding="utf-8-sig")
validation = {"input_sha256": data_sha256, "loaded_rows": loaded_rows, "scope_rows": len(raw),
    "analysis_rows": len(clean), "source_ids": clean.athlete_id.nunique(), "race_editions": clean.race_id.nunique(),
    "ambiguous_ids": len(ambiguous_ids), "identity_conflict_rows": len(identity_conflicts),
    "missing_age_rows": int(clean.age.isna().sum()), "filter_recovered_rows": len(missed),
    "quarantined_rows": len(quarantine), "max_speed_difference_kmh": float(clean.speed_difference.max()),
    "environment": {"Python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    "checks": "Row totals, grouping totals, date formats and speed-time reconciliation passed."}
(OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
print("Đối chiếu đạt. Đã lưu", len(tables), "bảng tổng hợp, dữ liệu phân tích và 4 biểu đồ vào results/.")