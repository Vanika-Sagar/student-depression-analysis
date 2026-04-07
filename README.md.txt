# Student Depression Dataset — Data Preprocessing & Insight Analysis

## Project Overview
This project was completed as part of the ACM SIGKDD R&D Domain recruitment task.
The goal was to take a real-world, intentionally messy dataset about student 
depression, clean it thoroughly, engineer meaningful features, visualize patterns, 
and extract data-backed insights without any model training.

**Dataset:** Student Depression Dataset (27,901 rows, 18 columns)  
**Tools Used:** Python, Pandas, NumPy, Matplotlib  
**Final Dataset:** 27,763 rows, 17 columns after cleaning


## Dataset Description
The dataset contains information about students across India including:

| Category | Columns |
| Demographics | Gender, Age, City |
| Academic | Academic Pressure, CGPA, Study Satisfaction, Degree |
| Lifestyle | Sleep Duration, Dietary Habits, Work/Study Hours |
| Mental Health | Depression, Suicidal Thoughts, Family History |
| Financial | Financial Stress |

**Target Variable:** `Depression` (0 = Not Depressed, 1 = Depressed)

## Step 1 — Data Understanding

The first step was to thoroughly explore the dataset before touching anything.

**Shape:** 27,901 rows and 18 columns  
**Key observations from df.info():**
- Only `Financial Stress` had missing values just 3 out of 27,901 (0.01%)
- `Age` was stored as float64 but should be integer
- 8 columns were text (str) type — needed inspection for inconsistencies
- `Work Pressure` and `Job Satisfaction` had suspiciously low means (~0.0004)

**Key observations from df.describe():**
- Age ranged from 18 to 59 - suspicious for a student dataset
- CGPA minimum was 0.0 — impossible in a real academic system
- Depression mean was 0.585 — meaning 58.5% of students are depressed
- Work Pressure and Job Satisfaction were nearly all zeros

**Text column inspection revealed:**
- Gender: Clean — only Male/Female
- Sleep Duration: 5 categories including vague 'Others'
- Dietary Habits: 4 categories including vague 'Others'
- Degree: 28 different degree types including 'Others'
- Suicidal Thoughts: Clean — only Yes/No
- Family History: Clean — only Yes/No
- Profession: 14 types — mix of students and working professionals
- City: **Severely corrupted** — contained degree names, people's names, 
  numbers and nonsense values mixed with real city names

---
## Step 2 — Data Cleaning & Preprocessing

### 2.1 Missing Values
- Ran `isnull().sum()` to detect explicit missing values
- Also checked for hidden missing values like '?', 'unknown', 'none', 'NA'
- **Result:** No hidden missing values found
- **Financial Stress** had 3 missing values — filled with **median (3.0)**
- **Decision:** Chose median over mean because median is unaffected by outliers.
  Chose to fill rather than drop because 3 rows out of 27,901 is only 0.01% —
  dropping would be wasteful.

### 2.2 City Column — Major Corruption Found
The City column was severely corrupted with invalid entries:
- Degree names appearing as cities: 'M.Tech', 'ME', 'M.Com'
- A number appearing as a city: '3.0'
- People's names as cities: 'Saanvi', 'Bhavna', 'Harsh', 'Gaurav' etc.
- Nonsense values: 'Less Delhi', 'Less than 5 Kalyan', 'City'

**Decision:** Replaced all invalid entries with NaN and dropped those rows.
These values cannot be reliably guessed or imputed — keeping them would 
introduce false data. Only 26 rows removed (0.09% of dataset).

### 2.3 Data Type Corrections
- Converted `Age` from float64 to int64 — age should be a whole number
- Dropped `id` column — purely a serial number with zero analytical value

### 2.4 Logical Errors
Several logically impossible or inconsistent entries were found and fixed:

| Issue | Count | Action | Reason |
| Age > 35 | 39 rows | Dropped | Not relevant for student analysis |
| CGPA = 0.0 | 8 rows | Dropped | Impossible in real academic system |
| Students with Work Pressure > 0 | 3 rows | Set to 0 | Students shouldn't have work pressure |
| Students with Job Satisfaction > 0 | 8 rows | Set to 0 | Students shouldn't have job satisfaction |

**Decision to drop Work Pressure and Job Satisfaction columns:**
99.9% of all values in both columns were 0 — this is almost entirely a student 
dataset. These columns carry no useful information and were dropped entirely.

### 2.5 Vague Category Cleanup
Small number of rows with vague/unclear categories were removed:
- 18 rows with 'Others' in Sleep Duration
- 12 rows with 'Others' in Dietary Habits  
- 35 rows with 'Others' in Degree

**Total rows removed in cleaning:** 138 (only 0.5% of original data)

---
## Step 3 — Feature Engineering

Five new meaningful columns were created from existing ones:

### 3.1 Age Group
**From:** Raw age values (18-35)  
**To:** 4 age bands — '18-20', '21-23', '24-27', '28-35'  
**Reason:** Age bands reveal generational patterns better than individual 
age values. Easier to compare and visualize groups.

| Age Group | Count |
| 18-20 | 5,378 |
| 21-23 | 4,524 |
| 24-27 | 6,654 |
| 28-35 | 11,272 |

### 3.2 Degree Level
**From:** 28 individual degree types  
**To:** 3 categories — Undergraduate, Postgraduate, Doctorate  
**Reason:** 28 categories is too granular for meaningful analysis. 
Broader groups reveal clearer patterns while retaining academic context.

| Degree Level | Count |
| Undergraduate | 19,365 |
| Postgraduate | 7,909 |
| Doctorate | 519 |

### 3.3 Is Student
**From:** 14 profession types  
**To:** Student / Professional  
**Reason:** Dataset is 99.9% students (27,797 out of 27,828). 
Simplifying profession into a binary column makes this clear.

### 3.4 Total Stress Score
**From:** Academic Pressure + Financial Stress  
**Formula:** (Academic Pressure + Financial Stress) / 2  
**Reason:** Both factors measure stress from different angles. 
Combining them creates a single composite stress indicator.  
**Range:** 0.5 to 5.0 | **Mean:** 3.14

### 3.5 Sleep Quality
**From:** Sleep Duration text categories  
**To:** Poor / Below Average / Good / Excessive  
**Reason:** Converts vague time ranges into meaningful quality labels
that are easier to interpret and visualize.

| Sleep Quality | Count |
| Poor (<5hrs) | 8,292 |
| Below Average (5-6hrs) | 6,161 |
| Good (7-8hrs) | 7,325 |
| Excessive (>8hrs) | 6,032 |

**Columns dropped after feature engineering:**
`Sleep Duration`, `Degree`, `Profession` — all replaced by better derived columns.

---
## Step 4 — Data Visualization & Analysis

### Chart 1 — Overall Depression Distribution
![Depression Distribution](chart1_depression_distribution.png)

**Finding:** 41.4% of students (11,522) are depressed while 58.6% (16,306) 
are not. Nearly 1 in 2 students in this dataset is depressed — an 
alarmingly high rate.

---

### Chart 2 — Depression Rate by Gender
![Depression by Gender](chart2_depression_by_gender.png)

**Finding:** Female depression rate (58.5%) and male depression rate (58.7%) 
are almost identical. Gender is NOT a significant differentiating factor 
in this dataset. The 0.2% difference is negligible.

---

### Chart 3 — Depression Rate by Sleep Quality
![Sleep vs Depression](chart3_sleep_vs_depression.png)

**Finding:** Students with poor sleep (<5 hours) have the highest depression 
rate at 64.5%. Depression generally decreases as sleep quality improves. 
Sleep deprivation is a strong indicator of depression.

---

### Chart 4 — Depression Rate by Academic Pressure
![Academic Pressure vs Depression](chart4_academic_pressure_vs_depression.png)

**Finding:** This is the strongest finding in the entire dataset. There is 
a near-perfect linear relationship between academic pressure and depression:
- Pressure 0 → 0.0% depression
- Pressure 5 → 86.1% depression

Academic pressure is the single strongest predictor of depression.

---

### Chart 5 — Depression Rate by Financial Stress
![Financial Stress vs Depression](chart5_financial_stress_vs_depression.png)

**Finding:** Financial stress shows a similarly strong linear relationship 
with depression. Students with highest financial stress (level 5) are 2.5x 
more likely to be depressed than those with lowest stress (level 1).
Financial stress is the second strongest predictor of depression.

---

### Chart 6 — Depression Rate by Age Group
![Age Group vs Depression](chart6_age_group_vs_depression.png)

**Finding:** Younger students are significantly more depressed. The 18-20 
age group has the highest depression rate at 72.3%. Depression consistently 
decreases as age increases — older students seem better equipped to handle 
academic life.

---

### Chart 7 — Depression Rate by Dietary Habits
![Dietary Habits vs Depression](chart7_dietary_habits_vs_depression.png)

**Finding:** Students with unhealthy dietary habits show significantly higher 
depression rates (70.8%) compared to those with healthy habits (45.4%). 
A 25.4% difference suggests a strong link between diet and mental health.

---

### Chart 8 — Correlation Heatmap
![Correlation Heatmap](chart8_correlation_heatmap.png)

**Key correlations with Depression:**
| Feature | Correlation |
| Total Stress | 0.55 (strongest) |
| Academic Pressure | 0.48 (very strong) |
| Financial Stress | 0.36 (moderate) |
| Work/Study Hours | 0.21 (weak) |
| Age | -0.23 (weak negative) |
| Study Satisfaction | -0.17 (weak negative) |
| CGPA | 0.02 (almost none) |

**Surprising finding:** CGPA has almost zero correlation with depression. 
Grades alone do not predict mental health.

---

### Chart 9 — Depression Rate by Family History
![Family History vs Depression](chart9_family_history_vs_depression.png)

**Finding:** Students with a family history of mental illness are somewhat 
more likely to be depressed (61.3% vs 56.0%). However the 5.3% difference 
is relatively small compared to academic and financial factors.

---

### Chart 10 — Depression Rate by Suicidal Thoughts
![Suicidal Thoughts vs Depression](chart10_suicidal_thoughts_vs_depression.png)

**Finding:** This is the most dramatic difference in the entire analysis. 
Students who have had suicidal thoughts are 3.4x more likely to be depressed 
(79.1% vs 23.2%). The 55.9% gap is the largest seen across all variables.

---

### Chart 11 — Depression Rate by Degree Level
![Degree Level vs Depression](chart11_degree_level_vs_depression.png)

**Finding:** Undergraduate students are the most depressed (60.8%), while 
postgraduate students show the lowest rate (53.3%). This aligns with the 
age finding — younger students who are mostly undergraduates face more 
depression.

---
## Step 5 — Key Insights

### Insight 1 — Academic Pressure is the Strongest Driver of Depression
Students with extreme academic pressure (level 5) have an 86.1% depression 
rate compared to 0% for those with no pressure. This perfect linear 
relationship makes academic pressure the single most important factor 
in this dataset.

### Insight 2 — Financial Stress is the Second Strongest Driver
Depression rises from 31.9% at low financial stress to 81.3% at high 
financial stress. Combined with academic pressure, financial burden 
creates an overwhelming environment for students.

### Insight 3 — Younger Students are Most Vulnerable
The 18-20 age group has a 72.3% depression rate — the highest of all 
groups. First year college students experiencing independence and academic 
pressure for the first time appear to be the most at-risk demographic.

### Insight 4 — Sleep and Diet are Lifestyle Indicators
Poor sleep (<5 hours) is associated with 64.5% depression rate while 
unhealthy diet is associated with 70.8% depression rate. These lifestyle 
factors are strongly linked to mental health outcomes.

### Insight 5 — CGPA Does Not Predict Depression
Surprisingly, CGPA has almost zero correlation with depression (0.02). 
This means a student with a 9.5 CGPA can be just as depressed as one 
with a 5.0 CGPA. Mental health cannot be judged by academic performance.

### Insight 6 — Suicidal Thoughts and Depression are Deeply Linked
79.1% of students who have had suicidal thoughts are depressed compared 
to only 23.2% of those who haven't. This 55.9% gap is the largest 
difference observed across all variables in the dataset.

### Insight 7 — Gender is Not a Factor
Depression affects male and female students almost equally (58.7% vs 
58.5%). Mental health struggles in academia are not gender specific.

### Insight 8 — Family History Plays a Minor Role
While students with family history of mental illness are slightly more 
depressed (61.3% vs 56.0%), the difference is small compared to 
environmental factors like academic and financial pressure. This suggests 
environment matters more than genetics in this dataset.

---

## Summary of Data Cleaning

| Step | Action | Rows/Columns Affected |
| Missing values | Filled Financial Stress with median | 3 rows filled |
| City corruption | Dropped invalid city entries | 26 rows removed |
| Data types | Converted Age to int, dropped id | 1 column removed |
| Logical errors | Dropped age>35, CGPA=0 | 47 rows removed |
| Redundant columns | Dropped Work Pressure, Job Satisfaction | 2 columns removed |
| Vague categories | Dropped Others/Unknown rows | 65 rows removed |
| Feature engineering | Created 5 new columns | 5 columns added |
| Redundant originals | Dropped Sleep Duration, Degree, Profession | 3 columns removed |

**Started with:** 27,901 rows, 18 columns  
**Ended with:** 27,763 rows, 17 columns  
**Data lost:** Only 0.5% of original data

---

## Tools Used
- **Python 3.13** — programming language
- **Pandas** — data manipulation and cleaning
- **NumPy** — mathematical operations
- **Matplotlib** — data visualization

---

*ACM SIGKDD R&D Domain Recruitment Task — Option 1: Data Preprocessing & 
Insight Analysis*