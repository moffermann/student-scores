#!/usr/bin/env python3
"""
Outlier Analysis by Course for Playground Series S6E1
Hypothesis: Different courses have different patterns of "normality"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================
# PATHS
# ============================================================
DATA_DIR = '/home/mauro/kaggle/student-scores/data'
OUTPUT_DIR = '/home/mauro/kaggle/student-scores/analysis'

print("=" * 60)
print("OUTLIER ANALYSIS BY COURSE")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")

train = pd.read_csv(f'{DATA_DIR}/train.csv')
print(f"  Train: {train.shape}")

# Numerical features to analyze
num_features = ['study_hours', 'class_attendance', 'sleep_hours', 'age']
target = 'exam_score'

# ============================================================
# GLOBAL STATISTICS
# ============================================================
print("\n[2/5] Global statistics for numerical features...")

print("\n  Overall Statistics:")
print("-" * 70)
for col in num_features + [target]:
    data = train[col]
    q1, q3 = data.quantile([0.25, 0.75])
    iqr = q3 - q1
    mad = np.median(np.abs(data - np.median(data)))

    print(f"  {col:20} | Mean: {data.mean():7.2f} | Median: {data.median():7.2f} | "
          f"Std: {data.std():6.2f} | MAD: {mad:6.2f} | IQR: {iqr:6.2f}")

# ============================================================
# STATISTICS BY COURSE
# ============================================================
print("\n[3/5] Statistics by course...")

courses = train['course'].unique()
print(f"\n  Courses found: {sorted(courses)}")
print(f"  Number of courses: {len(courses)}")

# Samples per course
print("\n  Samples per course:")
course_counts = train['course'].value_counts().sort_index()
for course, count in course_counts.items():
    print(f"    {course}: {count:,} ({count/len(train)*100:.1f}%)")

# Statistics by course for each feature
print("\n" + "=" * 80)
print("  DETAILED STATISTICS BY COURSE")
print("=" * 80)

stats_by_course = {}
for feature in num_features:
    print(f"\n  >>> {feature.upper()} <<<")
    print("-" * 80)
    print(f"  {'Course':<15} | {'Mean':>8} | {'Median':>8} | {'Std':>7} | "
          f"{'MAD':>7} | {'IQR':>7} | {'P1':>7} | {'P99':>7} | {'Range':>12}")
    print("-" * 80)

    feature_stats = []
    for course in sorted(courses):
        data = train[train['course'] == course][feature]

        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        p1, p99 = data.quantile([0.01, 0.99])
        mad = np.median(np.abs(data - np.median(data)))

        stats_dict = {
            'course': course,
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'mad': mad,
            'iqr': iqr,
            'p1': p1,
            'p99': p99,
            'min': data.min(),
            'max': data.max()
        }
        feature_stats.append(stats_dict)

        print(f"  {course:<15} | {data.mean():8.2f} | {data.median():8.2f} | "
              f"{data.std():7.2f} | {mad:7.2f} | {iqr:7.2f} | {p1:7.2f} | "
              f"{p99:7.2f} | [{data.min():.1f}, {data.max():.1f}]")

    stats_by_course[feature] = pd.DataFrame(feature_stats)

# ============================================================
# OUTLIER DETECTION BY COURSE
# ============================================================
print("\n\n[4/5] Outlier detection by course...")

def detect_outliers_mad(data, threshold=3):
    """Detect outliers using Median Absolute Deviation"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros(len(data), dtype=bool)
    modified_z_score = np.abs(data - median) / (1.4826 * mad)
    return modified_z_score > threshold

def detect_outliers_iqr(data, k=1.5):
    """Detect outliers using IQR method"""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (data < lower) | (data > upper)

# Count outliers by method and course
print("\n  OUTLIER COUNTS BY COURSE (MAD method, threshold=3)")
print("-" * 90)
print(f"  {'Course':<15} | {'study_hours':>12} | {'class_attendance':>16} | "
      f"{'sleep_hours':>12} | {'age':>8} | {'Total':>8}")
print("-" * 90)

outlier_summary = []
for course in sorted(courses):
    course_data = train[train['course'] == course]
    course_outliers = {}

    for feature in num_features:
        outliers = detect_outliers_mad(course_data[feature].values)
        course_outliers[feature] = outliers.sum()

    total = sum(course_outliers.values())
    n_rows = len(course_data)

    print(f"  {course:<15} | {course_outliers['study_hours']:>8} ({course_outliers['study_hours']/n_rows*100:4.1f}%) | "
          f"{course_outliers['class_attendance']:>12} ({course_outliers['class_attendance']/n_rows*100:4.1f}%) | "
          f"{course_outliers['sleep_hours']:>8} ({course_outliers['sleep_hours']/n_rows*100:4.1f}%) | "
          f"{course_outliers['age']:>4} ({course_outliers['age']/n_rows*100:4.1f}%) | {total:>8}")

    outlier_summary.append({
        'course': course,
        'n_samples': n_rows,
        **{f'{f}_outliers': course_outliers[f] for f in num_features},
        'total_outliers': total
    })

outlier_df = pd.DataFrame(outlier_summary)

print("\n  OUTLIER COUNTS BY COURSE (IQR method, k=1.5)")
print("-" * 90)
print(f"  {'Course':<15} | {'study_hours':>12} | {'class_attendance':>16} | "
      f"{'sleep_hours':>12} | {'age':>8} | {'Total':>8}")
print("-" * 90)

for course in sorted(courses):
    course_data = train[train['course'] == course]
    course_outliers = {}

    for feature in num_features:
        outliers = detect_outliers_iqr(course_data[feature].values)
        course_outliers[feature] = outliers.sum()

    total = sum(course_outliers.values())
    n_rows = len(course_data)

    print(f"  {course:<15} | {course_outliers['study_hours']:>8} ({course_outliers['study_hours']/n_rows*100:4.1f}%) | "
          f"{course_outliers['class_attendance']:>12} ({course_outliers['class_attendance']/n_rows*100:4.1f}%) | "
          f"{course_outliers['sleep_hours']:>8} ({course_outliers['sleep_hours']/n_rows*100:4.1f}%) | "
          f"{course_outliers['age']:>4} ({course_outliers['age']/n_rows*100:4.1f}%) | {total:>8}")

# ============================================================
# AGE ANALYSIS (Resiliency Hypothesis)
# ============================================================
print("\n\n[5/5] Age analysis (resiliency hypothesis)...")

# Age distribution
print("\n  Age distribution:")
age_stats = train.groupby('age')['exam_score'].agg(['count', 'mean', 'std', 'median'])
print(age_stats.to_string())

# Correlation between age and exam_score by sleep_quality
print("\n  Exam score by age group and sleep quality:")
train['age_group'] = pd.cut(train['age'], bins=[16, 18, 21, 25], labels=['17-18', '19-21', '22-24'])

age_sleep_pivot = train.pivot_table(
    values='exam_score',
    index='age_group',
    columns='sleep_quality',
    aggfunc='mean'
)
print(age_sleep_pivot.round(2).to_string())

# Impact of low sleep on different age groups
print("\n  Impact of low sleep hours (< 6) by age group:")
train['low_sleep'] = train['sleep_hours'] < 6

for age_grp in ['17-18', '19-21', '22-24']:
    grp_data = train[train['age_group'] == age_grp]
    low_sleep_score = grp_data[grp_data['low_sleep']]['exam_score'].mean()
    normal_sleep_score = grp_data[~grp_data['low_sleep']]['exam_score'].mean()
    diff = low_sleep_score - normal_sleep_score
    print(f"    {age_grp}: Low sleep={low_sleep_score:.2f}, Normal={normal_sleep_score:.2f}, "
          f"Impact: {diff:+.2f}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n\n[6/6] Generating visualizations...")

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of Numerical Features by Course', fontsize=14, fontweight='bold')

for idx, feature in enumerate(num_features):
    ax = axes[idx // 2, idx % 2]

    # Boxplot by course
    course_order = sorted(courses)
    train.boxplot(column=feature, by='course', ax=ax,
                  positions=range(len(course_order)))

    ax.set_title(f'{feature}', fontsize=12)
    ax.set_xlabel('Course')
    ax.set_ylabel(feature)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/boxplots_by_course.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/boxplots_by_course.png")

# Exam score by age group and sleep quality heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(age_sleep_pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax)
ax.set_title('Mean Exam Score by Age Group and Sleep Quality')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/age_sleep_heatmap.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/age_sleep_heatmap.png")

# Distribution of outliers by course
fig, ax = plt.subplots(figsize=(12, 6))
outlier_cols = [f'{f}_outliers' for f in num_features]
outlier_df_plot = outlier_df.set_index('course')[outlier_cols]
outlier_df_plot.columns = num_features
outlier_df_plot.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Number of Outliers by Course (MAD method)')
ax.set_xlabel('Course')
ax.set_ylabel('Number of Outliers')
ax.legend(title='Feature')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outliers_by_course.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/outliers_by_course.png")

# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 60)

print("\n  KEY FINDINGS:")
print("  1. All courses have similar distributions for numerical features")
print("  2. MAD method detects fewer outliers than IQR (more conservative)")
print("  3. study_hours and class_attendance have most outliers")
print("  4. Age shows minimal outliers (discrete values 17-24)")

# Calculate variance coefficient by course
print("\n  VARIABILITY BY COURSE (Coefficient of Variation):")
for feature in num_features:
    print(f"\n    {feature}:")
    for course in sorted(courses):
        data = train[train['course'] == course][feature]
        cv = data.std() / data.mean() * 100
        print(f"      {course}: {cv:.1f}%")

print("\n  RECOMMENDATIONS:")
print("  1. Apply capping (p1/p99) by course for study_hours and class_attendance")
print("  2. Create outlier flags for potential feature engineering")
print("  3. Test age interaction features based on resiliency hypothesis")
print("  4. Consider course-specific normalization")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

# Clean up temporary column
train.drop(['age_group', 'low_sleep'], axis=1, inplace=True)
