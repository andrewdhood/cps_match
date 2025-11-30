"""
SEHS Simulation v13.0 - Advanced Regional Preference Model
===========================================================

Key improvements over v12:
1. Score-based regional preference strength - high scorers more willing to travel
2. School-specific demand functions - each school has custom utility curves
3. Dynamic cross-region penalties based on student characteristics
4. Improved calibration for all 11 schools

This model aims for:
- Max error < 80 for all schools
- Per-school MAE < 25
- Overall MAE < 20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm
import warnings
from typing import Dict, List, Tuple, Callable

# Import centralized data (but override score distributions)
from sehs_data import (
    SCHOOLS, REGIONS, REGION_POPULATION, TIER_BY_REGION,
    REAL_DATA, ADMISSIONS,
    ELITE_SCHOOLS, CUTOFFS_2024_CALIBRATION,
    distance_miles, get_school
)

# V13 has its own score distributions, tuned specifically for this model
V13_SCORE_DISTRIBUTIONS = {
    # North Side - high scores for elite schools
    'north': {
        1: {'loc': 620, 'scale': 100, 'skew': 3.0},
        2: {'loc': 720, 'scale': 80, 'skew': 1.5},
        3: {'loc': 830, 'scale': 45, 'skew': -2.0},
        4: {'loc': 870, 'scale': 22, 'skew': -3.5},
    },
    # Loop - high scores for elite schools
    'loop': {
        1: {'loc': 700, 'scale': 90, 'skew': 2.5},
        2: {'loc': 755, 'scale': 75, 'skew': 1.0},
        3: {'loc': 835, 'scale': 40, 'skew': -2.0},
        4: {'loc': 872, 'scale': 20, 'skew': -3.5},
    },
    # West Side - OPTUNA TUNED (500 trials, trial 353, MAE 22.79)
    'west': {
        1: {'loc': 453, 'scale': 113, 'skew': 4.77},
        2: {'loc': 558, 'scale': 90, 'skew': 2.20},
        3: {'loc': 592, 'scale': 71, 'skew': 2.13},
        4: {'loc': 689, 'scale': 51, 'skew': 1.32},
    },
    # South Side - OPTUNA TUNED (500 trials, trial 353, MAE 22.79)
    'south': {
        1: {'loc': 457, 'scale': 100, 'skew': 4.04},
        2: {'loc': 541, 'scale': 87, 'skew': 3.31},
        3: {'loc': 612, 'scale': 79, 'skew': 0.58},
        4: {'loc': 724, 'scale': 47, 'skew': 0.96},
    },
}

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


# ============================================
# V13 MODEL PARAMETERS
# ============================================

PARAMS = {
    # Population
    'n_students': 22000,

    # Base friction coefficients (modified by score)
    'base_friction': {
        1: 3.0,   # T1 most distance-sensitive
        2: 2.2,
        3: 1.5,
        4: 1.0,   # T4 least sensitive
    },

    # Score-based mobility modifier
    # Higher scores = more willing to travel (lower friction)
    'score_mobility': {
        'threshold_high': 850,   # Above this, very mobile
        'threshold_mid': 750,    # Above this, moderately mobile
        'multiplier_high': 0.3,  # 70% friction reduction
        'multiplier_mid': 0.6,   # 40% friction reduction
        'multiplier_low': 1.0,   # No reduction
    },

    # Cross-region base penalties
    'cross_region_base': {
        ('north', 'south'): 50,
        ('north', 'west'): 35,
        ('south', 'north'): 25,  # Ambitious South students travel
        ('south', 'west'): 20,
        ('south', 'loop'): 5,
        ('west', 'north'): 30,
        ('west', 'south'): 25,
        ('west', 'loop'): 10,
        ('loop', 'south'): 15,
        ('loop', 'north'): 8,
        ('loop', 'west'): 12,
    },

    # Private school exit
    'exit_threshold': {'north': 892, 'loop': 894, 'west': 820, 'south': 800},
    'exit_prob': {'north': 0.18, 'loop': 0.22, 'west': 0.35, 'south': 0.40},

    # Distance cap
    'distance_cap': 15.0,
}


# ============================================
# SCHOOL-SPECIFIC DEMAND FUNCTIONS
# ============================================

def make_demand_function(school_name: str) -> Callable:
    """
    Create a school-specific demand function that returns utility modifier.

    Each function takes (tier, score, region) and returns utility adjustment.
    This allows fine-grained control over who applies to each school.
    """

    # Get actual cutoffs for this school
    cutoffs = CUTOFFS_2024_CALIBRATION.get(school_name, {})
    school = SCHOOLS.get(school_name)
    if not school:
        return lambda t, s, r: 0

    base_prestige = school.prestige

    if school_name == "Walter Payton":
        # Most competitive - only very high scorers apply
        def demand(tier, score, region):
            if tier == 4 and score < 880:
                return -30  # Don't bother if below T4 cutoff
            if tier == 3 and score < 860:
                return -25
            if score >= 895:
                return 15  # Bonus for top scorers
            return 0
        return demand

    elif school_name == "Northside":
        def demand(tier, score, region):
            if tier == 4 and score < 875:
                return -25
            if score >= 890:
                return 12
            if region == 'north':
                return 8  # Preference for local students
            return 0
        return demand

    elif school_name == "Whitney Young":
        def demand(tier, score, region):
            if tier == 4 and score < 865:
                return -20
            if region == 'loop':
                return 10  # Downtown preference
            if region == 'south' and score >= 850:
                return 5  # Accessible from South
            return 0
        return demand

    elif school_name == "Jones":
        def demand(tier, score, region):
            if tier == 4 and score < 850:
                return -18
            if region in ['loop', 'south']:
                return 8
            return 0
        return demand

    elif school_name == "Lane Tech":
        def demand(tier, score, region):
            if tier == 4 and score < 845:
                return -15
            if region == 'north':
                return 12  # Strong North preference
            if tier in [1, 2]:
                return 5  # Popular for lower tiers
            return 0
        return demand

    elif school_name == "Lindblom":
        # OPTUNA TUNED (500 trials, trial 353): lindblom_t4=-3
        def demand(tier, score, region):
            if tier == 4:
                return -3
            if region == 'south':
                return 5
            return 0
        return demand

    elif school_name == "King":
        # OPTUNA TUNED (500 trials, trial 353): king_t4_penalty=-24, king_other_penalty=-12
        def demand(tier, score, region):
            if tier == 4:
                return -24
            return -12
        return demand

    elif school_name == "Hancock":
        # OPTUNA TUNED (500 trials, trial 353): hancock_t4_penalty=-24, hancock_other_penalty=5
        def demand(tier, score, region):
            if tier == 4:
                return -24
            return 5
        return demand

    elif school_name == "Brooks":
        # OPTUNA TUNED (500 trials, trial 353): brooks_t4_penalty=-3, brooks_other_penalty=-12
        def demand(tier, score, region):
            if tier == 4:
                return -3
            return -12
        return demand

    elif school_name == "Westinghouse":
        # OPTUNA TUNED (500 trials, trial 353): westinghouse_non_west=-36, westinghouse_t34=-4
        def demand(tier, score, region):
            if region != 'west':
                return -36
            if tier in [3, 4]:
                return -4
            return -10
        return demand

    elif school_name == "South Shore":
        # OPTUNA TUNED (500 trials, trial 353): south_shore_t4=-10, south_shore_other=-9
        def demand(tier, score, region):
            if region in ['north', 'loop']:
                return -100
            if tier == 4:
                return -10
            return -9
        return demand

    else:
        return lambda t, s, r: 0


# Pre-compute demand functions for all schools
DEMAND_FUNCTIONS = {name: make_demand_function(name) for name in SCHOOLS.keys()}


# ============================================
# STUDENT GENERATION
# ============================================

def generate_students(n: int = None, seed: int = None) -> pd.DataFrame:
    """Generate student population with 4-region geography."""
    if seed is not None:
        np.random.seed(seed)

    if n is None:
        n = PARAMS['n_students']

    # Allocate to tiers
    tiers = np.random.choice([1, 2, 3, 4], size=n, p=[0.25, 0.25, 0.25, 0.25])

    # Allocate to regions
    regions = np.empty(n, dtype='<U10')
    for tier in [1, 2, 3, 4]:
        tier_mask = (tiers == tier)
        n_tier = np.sum(tier_mask)

        region_probs = TIER_BY_REGION[tier]
        region_names = list(region_probs.keys())
        region_weights = [region_probs[r] for r in region_names]

        regions[tier_mask] = np.random.choice(
            region_names, size=n_tier, p=region_weights
        )

    # Generate locations
    lats = np.zeros(n)
    lons = np.zeros(n)

    for region_name, region_data in REGIONS.items():
        mask = (regions == region_name)
        n_region = np.sum(mask)
        if n_region > 0:
            lats[mask] = np.random.normal(
                region_data['lat_center'], region_data['lat_std'], n_region
            )
            lons[mask] = np.random.normal(
                region_data['lon_center'], region_data['lon_std'], n_region
            )

    # Generate scores
    scores = np.zeros(n)
    for region_name in REGIONS.keys():
        for tier in [1, 2, 3, 4]:
            mask = (regions == region_name) & (tiers == tier)
            n_group = np.sum(mask)

            if n_group > 0:
                dist_params = V13_SCORE_DISTRIBUTIONS[region_name][tier]
                scores[mask] = skewnorm.rvs(
                    a=dist_params['skew'],
                    loc=dist_params['loc'],
                    scale=dist_params['scale'],
                    size=n_group
                )

    # Add noise and clip
    scores = np.clip(scores + np.random.normal(0, 3, n), 400, 900)

    # Create DataFrame
    df = pd.DataFrame({
        'Tier': tiers,
        'Region': regions,
        'Score': scores,
        'Lat': lats,
        'Lon': lons,
    })

    # Private school exit
    for region in REGIONS.keys():
        threshold = PARAMS['exit_threshold'].get(region, 850)
        prob = PARAMS['exit_prob'].get(region, 0.3)

        candidates = df[
            (df['Region'] == region) &
            (df['Tier'] == 4) &
            (df['Score'] > threshold)
        ].index

        if len(candidates) > 0:
            n_drop = int(len(candidates) * prob)
            drop_idx = np.random.choice(candidates, size=min(n_drop, len(candidates)), replace=False)
            df = df.drop(drop_idx)

    df['TieBreaker'] = np.random.random(len(df))
    return df.reset_index(drop=True)


# ============================================
# SCORE-BASED REGIONAL PREFERENCE
# ============================================

def get_mobility_multiplier(score: float) -> float:
    """
    Calculate mobility multiplier based on score.
    Higher scores = more willing to travel = lower friction.
    """
    if score >= PARAMS['score_mobility']['threshold_high']:
        return PARAMS['score_mobility']['multiplier_high']
    elif score >= PARAMS['score_mobility']['threshold_mid']:
        return PARAMS['score_mobility']['multiplier_mid']
    else:
        return PARAMS['score_mobility']['multiplier_low']


def get_friction(tier: int, score: float) -> float:
    """Get distance friction coefficient adjusted by score."""
    base = PARAMS['base_friction'].get(tier, 2.0)
    mobility = get_mobility_multiplier(score)
    return base * mobility


def get_cross_region_penalty(from_region: str, to_region: str, score: float, tier: int) -> float:
    """
    Calculate cross-region penalty, reduced for high scorers.
    """
    base_penalty = PARAMS['cross_region_base'].get((from_region, to_region), 30)

    # High scorers get reduced penalties (they're willing to travel)
    mobility = get_mobility_multiplier(score)

    # Elite-bound students (T4 with high scores) get extra mobility
    if tier == 4 and score >= 860:
        mobility *= 0.5

    return base_penalty * mobility


# ============================================
# PREFERENCE CALCULATION
# ============================================

def calculate_preferences(row: pd.Series) -> List[str]:
    """Calculate school preferences using v13 model features."""
    utilities = {}
    tier = row['Tier']
    region = row['Region']
    score = row['Score']
    student_lat = row['Lat']
    student_lon = row['Lon']

    friction = get_friction(tier, score)

    for school_name, school in SCHOOLS.items():
        # Calculate distance
        dist = distance_miles(student_lat, student_lon, school.lat, school.lon)

        # Skip if too far
        if dist > PARAMS['distance_cap']:
            continue

        # Base utility from prestige
        u = school.prestige

        # Distance penalty (score-adjusted)
        u -= dist * friction

        # Cross-region penalty (score-adjusted)
        school_region = school.region
        if region != school_region:
            penalty = get_cross_region_penalty(region, school_region, score, tier)
            u -= penalty

        # Draw region adjustment
        draw_weight = school.draw_regions.get(region, 0.05)
        if draw_weight < 0.10:
            u -= 15
        elif draw_weight < 0.20:
            u -= 8

        # School-specific demand function
        demand_fn = DEMAND_FUNCTIONS.get(school_name)
        if demand_fn:
            u += demand_fn(tier, score, region)

        # Skip if utility too low
        if u < 0:
            continue

        utilities[school_name] = u

    if len(utilities) == 0:
        return []

    # Sort by utility and return top 6
    ranked = sorted(utilities.keys(), key=lambda x: utilities[x], reverse=True)
    return ranked[:ADMISSIONS['max_choices']]


# ============================================
# MATCHING ALGORITHM
# ============================================

def run_match(student_df: pd.DataFrame) -> pd.DataFrame:
    """Run Serial Dictatorship matching with 30/70 split."""

    # Initialize seat counts
    seats = {}
    for name, school in SCHOOLS.items():
        seats[name] = {
            'Rank': int(school.seats * ADMISSIONS['rank_fraction']),
            1: int(school.seats * ADMISSIONS['tier_fraction']),
            2: int(school.seats * ADMISSIONS['tier_fraction']),
            3: int(school.seats * ADMISSIONS['tier_fraction']),
            4: int(school.seats * ADMISSIONS['tier_fraction']),
        }

    # Add preferences column
    student_df = student_df.copy()
    student_df['Preferences'] = student_df.apply(calculate_preferences, axis=1)
    student_df['Matched'] = None
    student_df['MatchType'] = None

    # Phase 1: Rank-based (top 30%)
    rank_sorted = student_df.sort_values(['Score', 'TieBreaker'], ascending=[False, True])

    for idx in rank_sorted.index:
        prefs = rank_sorted.loc[idx, 'Preferences']
        if len(prefs) == 0:
            continue

        for school in prefs:
            if seats[school]['Rank'] > 0:
                student_df.loc[idx, 'Matched'] = school
                student_df.loc[idx, 'MatchType'] = 'Rank'
                seats[school]['Rank'] -= 1
                break

    # Phase 2: Tier-based (remaining 70%)
    unmatched = student_df[student_df['Matched'].isna()]

    for tier in [1, 2, 3, 4]:
        tier_students = unmatched[unmatched['Tier'] == tier]
        tier_sorted = tier_students.sort_values(['Score', 'TieBreaker'], ascending=[False, True])

        for idx in tier_sorted.index:
            if student_df.loc[idx, 'Matched'] is not None:
                continue

            prefs = tier_sorted.loc[idx, 'Preferences']
            if len(prefs) == 0:
                continue

            for school in prefs:
                if seats[school][tier] > 0:
                    student_df.loc[idx, 'Matched'] = school
                    student_df.loc[idx, 'MatchType'] = 'Tier'
                    seats[school][tier] -= 1
                    break

    return student_df


# ============================================
# RESULTS COMPUTATION
# ============================================

def compute_cutoffs(student_df: pd.DataFrame) -> Dict:
    """Compute cutoff scores from matching results."""
    matched = student_df[student_df['Matched'].notna()]
    cutoffs = {}

    for school_name in SCHOOLS.keys():
        school_matched = matched[matched['Matched'] == school_name]
        cutoffs[school_name] = {}

        # Rank cutoff
        rank_matched = school_matched[school_matched['MatchType'] == 'Rank']
        if len(rank_matched) > 0:
            cutoffs[school_name]['Rank'] = rank_matched['Score'].min()

        # Tier cutoffs
        tier_matched = school_matched[school_matched['MatchType'] == 'Tier']
        for tier in [1, 2, 3, 4]:
            tier_students = tier_matched[tier_matched['Tier'] == tier]
            if len(tier_students) > 0:
                cutoffs[school_name][tier] = tier_students['Score'].min()

    return cutoffs


def compute_metrics(cutoffs: Dict) -> Dict:
    """Compute error metrics against actual cutoffs."""
    errors = []
    school_errors = {}

    for school_name in SCHOOLS.keys():
        actual = CUTOFFS_2024_CALIBRATION.get(school_name, {})
        simulated = cutoffs.get(school_name, {})

        school_err = []
        max_err = 0
        worst_tier = None

        for tier in ['Rank', 1, 2, 3, 4]:
            if tier in actual and tier in simulated:
                err = simulated[tier] - actual[tier]
                school_err.append(abs(err))
                errors.append(abs(err))
                if abs(err) > abs(max_err):
                    max_err = err
                    worst_tier = tier

        school_errors[school_name] = {
            'mae': np.mean(school_err) if school_err else 0,
            'max_err': max_err,
            'worst_tier': worst_tier,
        }

    return {
        'overall_mae': np.mean(errors) if errors else 0,
        'overall_rmse': np.sqrt(np.mean([e**2 for e in errors])) if errors else 0,
        'max_error': max(errors) if errors else 0,
        'school_errors': school_errors,
    }


# ============================================
# SIMULATION RUNNER
# ============================================

def run_simulation(seed: int = None, verbose: bool = True) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Run complete simulation and return results."""

    if verbose:
        print("SEHS Simulation v13.0 - Advanced Regional Preference Model")
        print("=" * 60)

    # Generate students
    if verbose:
        print("Generating students...")
    students = generate_students(seed=seed)

    if verbose:
        print(f"  Total students: {len(students):,}")
        for region in ['north', 'loop', 'west', 'south']:
            n = len(students[students['Region'] == region])
            print(f"    {region.capitalize():<8}: {n:,}")

    # Run matching
    if verbose:
        print("Calculating preferences...")
        print("Running matching algorithm...")

    matched_students = run_match(students)

    # Compute cutoffs and metrics
    cutoffs = compute_cutoffs(matched_students)
    metrics = compute_metrics(cutoffs)

    # Create results DataFrame
    results_list = []
    for school_name, school_cutoffs in cutoffs.items():
        for tier, score in school_cutoffs.items():
            matched = matched_students[
                (matched_students['Matched'] == school_name) &
                ((matched_students['MatchType'] == 'Rank') if tier == 'Rank' else
                 ((matched_students['MatchType'] == 'Tier') & (matched_students['Tier'] == tier)))
            ]
            for _, row in matched.iterrows():
                results_list.append({
                    'School': school_name,
                    'Type': tier,
                    'Score': row['Score'],
                    'Tier': row['Tier'],
                    'Region': row['Region'],
                })

    results_df = pd.DataFrame(results_list)

    if verbose:
        print()
        print("=" * 70)
        print("SIMULATION RESULTS v13.0")
        print("=" * 70)
        print()
        print("OVERALL METRICS:")
        print(f"  MAE:  {metrics['overall_mae']:.1f}")
        print(f"  RMSE: {metrics['overall_rmse']:.1f}")
        print(f"  Max Error: {metrics['max_error']:.0f}")
        print()

        # Elite vs Other
        elite_errors = [metrics['school_errors'][s]['mae'] for s in ELITE_SCHOOLS]
        other_errors = [metrics['school_errors'][s]['mae'] for s in SCHOOLS if s not in ELITE_SCHOOLS]
        print(f"  Elite Schools MAE: {np.mean(elite_errors):.1f}")
        print(f"  Other Schools MAE: {np.mean(other_errors):.1f}")
        print()

        # Per-school breakdown
        print(f"{'School':<18} {'MAE':>6} {'MaxErr':>10} {'Worst':>8}")
        print("-" * 50)

        sorted_schools = sorted(
            metrics['school_errors'].items(),
            key=lambda x: x[1]['mae'],
            reverse=True
        )

        for school_name, err in sorted_schools:
            mae = err['mae']
            max_err = err['max_err']
            worst = err['worst_tier']
            flag = " ***" if mae > 25 or abs(max_err) > 80 else ""
            print(f"{school_name:<18} {mae:>6.1f} {max_err:>+10.0f} {str(worst):>8}{flag}")

        print()

        # Regional admits
        print("ADMITS BY REGION:")
        matched = matched_students[matched_students['Matched'].notna()]
        for region in ['north', 'loop', 'west', 'south']:
            region_matched = matched[matched['Region'] == region]
            elite_matched = region_matched[region_matched['Matched'].isin(ELITE_SCHOOLS)]
            print(f"  {region.capitalize():<8}: {len(region_matched):>5} total, {len(elite_matched):>5} elite")

    return matched_students, results_df, metrics


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run with multiple seeds for validation
    print()
    students, results, metrics = run_simulation(seed=42, verbose=True)

    print()
    print("=" * 60)
    print("Multi-seed validation:")
    print("=" * 60)

    maes = []
    for seed in [42, 123, 456, 789, 1000]:
        _, _, m = run_simulation(seed=seed, verbose=False)
        maes.append(m['overall_mae'])
        print(f"  Seed {seed}: MAE = {m['overall_mae']:.1f}")

    print()
    print(f"  Average MAE: {np.mean(maes):.1f} (+/- {np.std(maes):.1f})")
    print()

    # Check targets
    max_school_mae = max(metrics['school_errors'][s]['mae'] for s in SCHOOLS)
    max_error = metrics['max_error']

    print("TARGET CHECK:")
    print(f"  Max school MAE: {max_school_mae:.1f} (target: < 25) {'✓' if max_school_mae < 25 else '✗'}")
    print(f"  Max error: {max_error:.0f} (target: < 80) {'✓' if max_error < 80 else '✗'}")
