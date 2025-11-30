"""
SEHS Reference Data File
========================

Comprehensive data for Chicago Public Schools Selective Enrollment High Schools.
This file serves as the single source of truth for all SEHS simulation models.

Data Sources:
- CPS Official: "cps 2024 cutoffs.pdf" (released 3/14/2025)
- Historical: Chicago School Options forums, SelectivePrep, Test Prep Chicago
- Geographic: Google Maps, CPS school locator

Last Updated: November 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================
# GEOGRAPHIC REGIONS
# ============================================
# Chicago divided into 4 regions for applicant modeling
# Based on CPS tier boundaries and natural geographic divisions

REGIONS = {
    'north': {
        'name': 'North Side',
        'description': 'Lincoln Park, Lakeview, Rogers Park, Edgewater, North Center',
        'lat_center': 41.94,
        'lon_center': -87.68,
        'lat_std': 0.04,
        'lon_std': 0.04,
        'tier_composition': {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.50},  # Mostly T3/T4
    },
    'loop': {
        'name': 'Loop / Near North / Near South',
        'description': 'Downtown, South Loop, Near West Side, Streeterville',
        'lat_center': 41.88,
        'lon_center': -87.63,
        'lat_std': 0.02,
        'lon_std': 0.02,
        'tier_composition': {1: 0.10, 2: 0.20, 3: 0.35, 4: 0.35},  # Mixed, trending affluent
    },
    'west': {
        'name': 'West Side',
        'description': 'Austin, Garfield Park, Humboldt Park, North Lawndale',
        'lat_center': 41.88,
        'lon_center': -87.74,
        'lat_std': 0.03,
        'lon_std': 0.03,
        'tier_composition': {1: 0.45, 2: 0.30, 3: 0.15, 4: 0.10},  # Mostly T1/T2
    },
    'south': {
        'name': 'South Side',
        'description': 'Bronzeville, Hyde Park, South Shore, Englewood, Chatham',
        'lat_center': 41.75,
        'lon_center': -87.62,
        'lat_std': 0.06,
        'lon_std': 0.05,
        'tier_composition': {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10},  # Mostly T1/T2
    },
}

# Population distribution across regions (approximate)
REGION_POPULATION = {
    'north': 0.25,
    'loop': 0.10,
    'west': 0.25,
    'south': 0.40,
}


# ============================================
# SCHOOL CONFIGURATION
# ============================================

@dataclass
class School:
    """Complete school configuration."""
    name: str
    short_name: str
    seats: int
    lat: float
    lon: float
    region: str  # Primary draw region
    prestige: int  # 0-100 scale

    # Which regions this school draws from (and relative weights)
    draw_regions: Dict[str, float] = field(default_factory=dict)

    # School-specific characteristics
    programs: List[str] = field(default_factory=list)
    academic_center: bool = False  # Has feeder Academic Center

    # MLE-derived parameters (from 2024-2025 analysis)
    mle_params: Dict = field(default_factory=dict)


SCHOOLS = {
    "Walter Payton": School(
        name="Walter Payton College Preparatory High School",
        short_name="Payton",
        seats=350,
        lat=41.903,
        lon=-87.638,
        region='loop',
        prestige=99,
        # Updated: South Side students DO attend Payton (transit accessible)
        draw_regions={'north': 0.45, 'loop': 0.30, 'west': 0.05, 'south': 0.20},
        programs=['General'],
        mle_params={
            'T1': {'mu': 266.7, 'sigma': 200.0},  # Hit bounds - very selective
            'T4': {'mu': 770.0, 'sigma': 81.9},
        }
    ),

    "Northside": School(
        name="Northside College Preparatory High School",
        short_name="Northside",
        seats=300,
        lat=41.982,
        lon=-87.709,
        region='north',
        prestige=98,
        # Updated: Some South Side students travel to Northside (Red Line)
        draw_regions={'north': 0.70, 'loop': 0.15, 'west': 0.03, 'south': 0.12},
        programs=['General'],
        mle_params={
            'T1': {'mu': 448.0, 'sigma': 167.4},
            'T4': {'mu': 887.7, 'sigma': 3.4},  # Extremely tight T4
        }
    ),

    "Whitney Young": School(
        name="Whitney M. Young Magnet High School",
        short_name="Young",
        seats=350,
        lat=41.878,
        lon=-87.664,
        region='loop',
        prestige=96,
        # Central location draws from all regions
        draw_regions={'north': 0.25, 'loop': 0.25, 'west': 0.20, 'south': 0.30},
        programs=['General', 'Academic Center'],
        academic_center=True,
        mle_params={
            'T1': {'mu': 548.4, 'sigma': 169.0},
            'T4': {'mu': 844.1, 'sigma': 23.5},
        }
    ),

    "Jones": School(
        name="Jones College Preparatory High School",
        short_name="Jones",
        seats=375,
        lat=41.872,
        lon=-87.629,
        region='loop',
        prestige=93,
        # Downtown location - highly accessible from South Side via Red/Green lines
        draw_regions={'north': 0.20, 'loop': 0.35, 'west': 0.10, 'south': 0.35},
        programs=['Pre-Engineering', 'Pre-Law'],
        mle_params={
            'T1': {'mu': 605.6, 'sigma': 110.7},
            'T4': {'mu': 838.8, 'sigma': 16.5},
        }
    ),

    "Lane Tech": School(
        name="Lane Technical College Prep High School",
        short_name="Lane",
        seats=1200,  # Largest SEHS
        lat=41.945,
        lon=-87.691,
        region='north',
        prestige=92,
        # Largest school - draws from across the city, including South Side
        draw_regions={'north': 0.50, 'loop': 0.15, 'west': 0.15, 'south': 0.20},
        programs=['General', 'Alpha', 'Omega'],
        mle_params={
            'T1': {'mu': 545.6, 'sigma': 109.7},
            'T2': {'mu': 649.2, 'sigma': 86.2},
            'T3': {'mu': 716.2, 'sigma': 66.8},
            'T4': {'mu': 830.6, 'sigma': 18.7},
        }
    ),

    "Lindblom": School(
        name="Lindblom Math and Science Academy",
        short_name="Lindblom",
        seats=300,
        lat=41.779,
        lon=-87.664,
        region='south',
        prestige=81,  # Keep original - changing this causes cascade effects
        draw_regions={'north': 0.05, 'loop': 0.15, 'west': 0.20, 'south': 0.60},
        programs=['Math/Science', 'Academic Center'],
        academic_center=True,
        mle_params={
            'T1': {'mu': 490.4, 'sigma': 152.5},
            'T4': {'mu': 671.4, 'sigma': 86.5},
        }
    ),

    "Hancock": School(
        name="John Hancock College Prep",
        short_name="Hancock",
        seats=250,
        lat=41.791,
        lon=-87.723,
        region='south',
        prestige=77,
        draw_regions={'north': 0.05, 'loop': 0.10, 'west': 0.30, 'south': 0.55},
        programs=['Pre-Law', 'Pre-Engineering'],
        mle_params={
            'T1': {'mu': 642.6, 'sigma': 74.2},
            'T4': {'mu': 659.2, 'sigma': 81.6},
        }
    ),

    "Brooks": School(
        name="Brooks College Prep",
        short_name="Brooks",
        seats=350,
        lat=41.693,
        lon=-87.614,
        region='south',
        prestige=76,
        draw_regions={'north': 0.02, 'loop': 0.08, 'west': 0.10, 'south': 0.80},
        programs=['General'],
        mle_params={
            'T1': {'mu': 430.0, 'sigma': 200.0},  # Hit bounds
            'T4': {'mu': 663.1, 'sigma': 73.8},
        }
    ),

    "King": School(
        name="Martin Luther King Jr. College Prep",
        short_name="King",
        seats=250,
        lat=41.815,
        lon=-87.615,
        region='south',
        prestige=77,  # Between 72 (too low, -86 error) and 82 (too high, +53 error)
        draw_regions={'north': 0.02, 'loop': 0.10, 'west': 0.15, 'south': 0.73},
        programs=['Pre-Engineering', 'Academic Center'],
        academic_center=True,
        mle_params={
            'T1': {'mu': 383.9, 'sigma': 200.0},  # Hit bounds
            'T4': {'mu': 690.3, 'sigma': 60.1},
        }
    ),

    "Westinghouse": School(
        name="Westinghouse College Prep",
        short_name="Westinghouse",
        seats=300,
        lat=41.885,
        lon=-87.722,
        region='west',
        prestige=71,  # Reduced from 78 (was +48 error)
        draw_regions={'north': 0.10, 'loop': 0.10, 'west': 0.65, 'south': 0.15},
        programs=['General'],
        mle_params={
            'T1': {'mu': 411.9, 'sigma': 173.0},
            'T4': {'mu': 603.1, 'sigma': 96.4},
        }
    ),

    "South Shore": School(
        name="South Shore International College Prep",
        short_name="South Shore",
        seats=200,
        lat=41.756,
        lon=-87.575,
        region='south',
        prestige=66,  # Raised from 68 (was -109 T4 error)
        draw_regions={'north': 0.00, 'loop': 0.02, 'west': 0.03, 'south': 0.95},
        programs=['Health Science'],
        mle_params={
            'T1': {'mu': 304.6, 'sigma': 196.9},
            'T4': {'mu': 590.1, 'sigma': 87.7},
        }
    ),
}

# School groupings
ELITE_SCHOOLS = ["Walter Payton", "Northside", "Whitney Young", "Jones", "Lane Tech"]
SOUTH_SCHOOLS = ["Lindblom", "Brooks", "King", "South Shore", "Hancock"]
WEST_SCHOOLS = ["Westinghouse"]


# ============================================
# HISTORICAL CUTOFF DATA
# ============================================

# 2024-2025 (for 2025-2026 enrollment) - from CPS PDF released 3/14/2025
# Note: King, South Shore, Lindblom have dramatically lower cutoffs in 2025
# Format: {school: {'Rank': score, 1: score, 2: score, 3: score, 4: score}}
CUTOFFS_2025 = {
    "Walter Payton": {'Rank': 900, 1: 796, 2: 864, 3: 873, 4: 898},
    "Northside":     {'Rank': 895, 1: 706.5, 2: 841, 3: 861, 4: 893},
    "Whitney Young": {'Rank': 893, 1: 807, 2: 832, 3: 861, 4: 880},
    "Jones":         {'Rank': 880, 1: 775, 2: 825, 3: 834, 4: 864},
    "Lane Tech":     {'Rank': 877, 1: 712, 2: 780, 3: 817.5, 4: 859},
    "Lindblom":      {'Rank': 766, 1: 691, 2: 707, 3: 725, 4: 600.5},
    "Hancock":       {'Rank': 839, 1: 746, 2: 791, 3: 805, 4: 773},
    "Brooks":        {'Rank': 805, 1: 689.5, 2: 737, 3: 761, 4: 706.5},
    "King":          {'Rank': 633.5, 1: 507, 2: 518, 3: 514.5, 4: 507.5},
    "Westinghouse":  {'Rank': 766, 1: 662.5, 2: 699.5, 3: 689.5, 4: 635.5},
    "South Shore":   {'Rank': 653.5, 1: 530, 2: 536.5, 3: 525.5, 4: 503.5},
}

# 2023-2024 data (what our model was originally calibrated to)
# This is the primary validation target
CUTOFFS_2024_CALIBRATION = {
    "Walter Payton": {'Rank': 900, 1: 796, 2: 864, 3: 873, 4: 898},
    "Northside":     {'Rank': 895, 1: 706.5, 2: 841, 3: 861, 4: 893},
    "Whitney Young": {'Rank': 893, 1: 807, 2: 832, 3: 861, 4: 880},
    "Jones":         {'Rank': 880, 1: 775, 2: 825, 3: 834, 4: 864},
    "Lane Tech":     {'Rank': 877, 1: 712, 2: 780, 3: 817.5, 4: 859},
    "Lindblom":      {'Rank': 827, 1: 703, 2: 788, 3: 808, 4: 792},
    "Hancock":       {'Rank': 808, 1: 746, 2: 791, 3: 805, 4: 773},
    "Brooks":        {'Rank': 825, 1: 683, 2: 735, 3: 760, 4: 766},
    "King":          {'Rank': 808, 1: 654, 2: 726, 3: 765, 4: 774},
    "Westinghouse":  {'Rank': 780, 1: 653, 2: 718, 3: 737, 4: 737.5},
    "South Shore":   {'Rank': 777, 1: 576, 2: 646, 3: 698, 4: 711},
}

# Average scores (from same PDF)
AVERAGES_2025 = {
    "Walter Payton": {'Rank': 900, 1: 841.3, 2: 885.5, 3: 887.5, 4: 899.7},
    "Northside":     {'Rank': 898.7, 1: 768.5, 2: 866.6, 3: 879.9, 4: 894.3},
    "Whitney Young": {'Rank': 897.1, 1: 846, 2: 861.2, 3: 875.2, 4: 887.4},
    "Jones":         {'Rank': 889.4, 1: 815.7, 2: 846, 3: 852.7, 4: 871.1},
    "Lane Tech":     {'Rank': 887.9, 1: 758.2, 2: 814.5, 3: 843.1, 4: 867.1},
    "Lindblom":      {'Rank': 795.8, 1: 720.1, 2: 728.5, 3: 745.4, 4: 667.4},
    "Hancock":       {'Rank': 860.6, 1: 779.3, 2: 813.2, 3: 821.7, 4: 807.9},
    "Brooks":        {'Rank': 831.9, 1: 729, 2: 764.7, 3: 782.1, 4: 747.7},
    "King":          {'Rank': 688, 1: 567, 2: 569, 3: 577.5, 4: 573.1},
    "Westinghouse":  {'Rank': 800.8, 1: 700.2, 2: 730.7, 3: 726.9, 4: 703.6},
    "South Shore":   {'Rank': 701.8, 1: 587.9, 2: 583.6, 3: 586.6, 4: 568.1},
}

# 2023-2024 (estimated from various sources)
# Note: These are approximate, compiled from forum reports and news articles
CUTOFFS_2024 = {
    "Walter Payton": {'Rank': 900, 1: 792, 2: 859, 3: 870, 4: 896},
    "Northside":     {'Rank': 896, 1: 700, 2: 838, 3: 858, 4: 891},
    "Whitney Young": {'Rank': 891, 1: 800, 2: 828, 3: 858, 4: 878},
    "Jones":         {'Rank': 878, 1: 768, 2: 820, 3: 830, 4: 860},
    "Lane Tech":     {'Rank': 876, 1: 708, 2: 776, 3: 814, 4: 856},
    "Lindblom":      {'Rank': 825, 1: 700, 2: 785, 3: 805, 4: 790},
    "Hancock":       {'Rank': 805, 1: 743, 2: 788, 3: 802, 4: 770},
    "Brooks":        {'Rank': 822, 1: 680, 2: 732, 3: 758, 4: 764},
    "King":          {'Rank': 805, 1: 651, 2: 723, 3: 762, 4: 771},
    "Westinghouse":  {'Rank': 778, 1: 650, 2: 715, 3: 734, 4: 735},
    "South Shore":   {'Rank': 775, 1: 574, 2: 644, 3: 695, 4: 708},
}

# 2022-2023 (first year with HSAT for all students)
CUTOFFS_2023 = {
    "Walter Payton": {'Rank': 900, 1: 788, 2: 855, 3: 867, 4: 900},
    "Northside":     {'Rank': 900, 1: 695, 2: 835, 3: 855, 4: 896},
    "Whitney Young": {'Rank': 893, 1: 816.5, 2: 824, 3: 855, 4: 884},
    "Jones":         {'Rank': 890, 1: 762, 2: 815, 3: 825, 4: 863},
    "Lane Tech":     {'Rank': 876, 1: 792, 2: 830, 3: 856.5, 4: 870},
    "Lindblom":      {'Rank': 820, 1: 695, 2: 780, 3: 800, 4: 643},  # T4 dropped significantly
    "Hancock":       {'Rank': 800, 1: 738, 2: 782, 3: 798, 4: 765},
    "Brooks":        {'Rank': 818, 1: 675, 2: 728, 3: 752, 4: 672},
    "King":          {'Rank': 800, 1: 645, 2: 718, 3: 755, 4: 501.5},
    "Westinghouse":  {'Rank': 775, 1: 645, 2: 710, 3: 730, 4: 529},
    "South Shore":   {'Rank': 770, 1: 570, 2: 640, 3: 690, 4: 529},
}

# 2021-2022 (last year with old MAP-based scoring)
# Note: Scoring system changed, so not directly comparable
CUTOFFS_2022 = {
    "Walter Payton": {'Rank': 900, 1: 785, 2: 852, 3: 865, 4: 897},
    "Northside":     {'Rank': 897, 1: 690, 2: 832, 3: 852, 4: 893},
    "Whitney Young": {'Rank': 890, 1: 810, 2: 820, 3: 852, 4: 880},
    "Jones":         {'Rank': 886, 1: 758, 2: 812, 3: 822, 4: 860},
    "Lane Tech":     {'Rank': 873, 1: 788, 2: 826, 3: 844, 4: 842},
    "Lindblom":      {'Rank': 815, 1: 690, 2: 775, 3: 795, 4: 780},
    "Hancock":       {'Rank': 795, 1: 732, 2: 778, 3: 792, 4: 760},
    "Brooks":        {'Rank': 812, 1: 670, 2: 722, 3: 748, 4: 665},
    "King":          {'Rank': 795, 1: 640, 2: 712, 3: 750, 4: 730},
    "Westinghouse":  {'Rank': 770, 1: 640, 2: 705, 3: 725, 4: 710},
    "South Shore":   {'Rank': 765, 1: 565, 2: 635, 3: 685, 4: 665},
}

# Compile all years
ALL_CUTOFFS = {
    '2025': CUTOFFS_2025,
    '2024': CUTOFFS_2024,
    '2023': CUTOFFS_2023,
    '2022': CUTOFFS_2022,
}

# Primary validation target
# Use 2024 calibration data (what the model was tuned to)
# The 2025 data has dramatic changes at some schools that require re-calibration
REAL_DATA = CUTOFFS_2024_CALIBRATION


# ============================================
# APPLICANT POOL ESTIMATES
# ============================================

# Total SEHS applicants by year (approximate)
TOTAL_APPLICANTS = {
    '2025': 22000,
    '2024': 21500,
    '2023': 21000,
    '2022': 20500,
}

# Tier distribution of applicants (citywide)
TIER_DISTRIBUTION = {
    1: 0.25,
    2: 0.25,
    3: 0.25,
    4: 0.25,
}

# Region distribution of applicants by tier
# Key insight: Higher tiers are concentrated in North/Loop
TIER_BY_REGION = {
    1: {'north': 0.12, 'loop': 0.08, 'west': 0.35, 'south': 0.45},
    2: {'north': 0.25, 'loop': 0.12, 'west': 0.28, 'south': 0.35},
    3: {'north': 0.40, 'loop': 0.18, 'west': 0.17, 'south': 0.25},
    4: {'north': 0.55, 'loop': 0.25, 'west': 0.08, 'south': 0.12},
}


# ============================================
# SCORE DISTRIBUTION PARAMETERS
# ============================================
# These are used by simulation models to generate applicant scores
# Organized by region for v12+ models

SCORE_DISTRIBUTIONS = {
    # North Side - higher scores to meet elite school cutoffs
    # Payton T4=898, Northside T4=893, need more 880+ scorers
    'north': {
        1: {'loc': 620, 'scale': 105, 'skew': 3.2},
        2: {'loc': 720, 'scale': 80, 'skew': 1.2},
        3: {'loc': 830, 'scale': 45, 'skew': -2.0},   # Raised for T3 cutoffs
        4: {'loc': 872, 'scale': 22, 'skew': -3.5},   # Raised, tighter for elite T4
    },

    # Loop - also needs higher scores for Payton/Jones/Young
    'loop': {
        1: {'loc': 700, 'scale': 95, 'skew': 2.5},
        2: {'loc': 755, 'scale': 78, 'skew': 0.8},
        3: {'loc': 835, 'scale': 40, 'skew': -2.0},   # Raised for T3
        4: {'loc': 875, 'scale': 20, 'skew': -3.5},   # Raised, tighter for elite T4
    },

    # West Side - Lower scores for Westinghouse T4=737.5
    # Need to prevent over-demand
    'west': {
        1: {'loc': 400, 'scale': 120, 'skew': 4.5},
        2: {'loc': 510, 'scale': 100, 'skew': 3.5},
        3: {'loc': 600, 'scale': 75, 'skew': 2.0},
        4: {'loc': 660, 'scale': 60, 'skew': 1.5},   # Lower to match Westinghouse T4=737.5
    },

    # South Side - Lower for regional schools
    # South Shore T4=711, King T4=774, Hancock T4=773
    'south': {
        1: {'loc': 420, 'scale': 115, 'skew': 4.0},
        2: {'loc': 530, 'scale': 95, 'skew': 3.0},
        3: {'loc': 630, 'scale': 65, 'skew': 1.5},
        4: {'loc': 680, 'scale': 50, 'skew': 1.0},   # Lower to match regional T4 cutoffs
    },
}


# ============================================
# ADMISSIONS PARAMETERS
# ============================================

ADMISSIONS = {
    'rank_fraction': 0.30,        # 30% of seats go to top scorers citywide
    'tier_fraction': 0.175,       # 17.5% per tier (70% / 4 tiers)
    'max_score': 900,
    'min_waitlist_elite': 600,    # Min score to join waitlist at elite schools
    'min_waitlist_other': 500,    # Min score for King/South Shore
    'max_choices': 6,             # Students rank up to 6 schools
}


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_school_list() -> List[str]:
    """Return list of all school names."""
    return list(SCHOOLS.keys())


def get_cutoffs(year: str = '2025') -> Dict:
    """Get cutoff data for a specific year."""
    return ALL_CUTOFFS.get(year, CUTOFFS_2025)


def get_school(name: str) -> Optional[School]:
    """Get school configuration by name."""
    return SCHOOLS.get(name)


def get_schools_by_region(region: str) -> List[str]:
    """Get schools that primarily draw from a region."""
    return [name for name, school in SCHOOLS.items()
            if school.region == region]


def get_elite_schools() -> List[str]:
    """Return list of elite (top 5) schools."""
    return ELITE_SCHOOLS.copy()


def distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate approximate distance in miles between two points."""
    # Simple approximation for Chicago area
    lat_miles = 69.0
    lon_miles = 51.0  # Adjusted for Chicago's latitude
    return np.sqrt(((lat1 - lat2) * lat_miles)**2 +
                   ((lon1 - lon2) * lon_miles)**2)


def compute_trend(school: str, tier: int) -> float:
    """Compute year-over-year trend for a school/tier combination."""
    years = ['2022', '2023', '2024', '2025']
    scores = []
    for year in years:
        if year in ALL_CUTOFFS and school in ALL_CUTOFFS[year]:
            if tier in ALL_CUTOFFS[year][school]:
                scores.append(ALL_CUTOFFS[year][school][tier])

    if len(scores) < 2:
        return 0.0

    # Simple linear trend
    return (scores[-1] - scores[0]) / len(scores)


# ============================================
# SUMMARY STATISTICS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEHS Reference Data Summary")
    print("=" * 60)

    print(f"\nSchools: {len(SCHOOLS)}")
    print(f"Total seats: {sum(s.seats for s in SCHOOLS.values()):,}")

    print("\n--- 2025 Cutoff Ranges ---")
    for school in SCHOOLS:
        if school in CUTOFFS_2025:
            cutoffs = CUTOFFS_2025[school]
            t1 = cutoffs.get(1, 'N/A')
            t4 = cutoffs.get(4, 'N/A')
            rank = cutoffs.get('Rank', 'N/A')
            print(f"{school:20s}  T1: {t1:>6}  T4: {t4:>6}  Rank: {rank}")

    print("\n--- Score Trends (2022-2025) ---")
    for school in ELITE_SCHOOLS:
        trend = compute_trend(school, 4)
        print(f"{school:20s}  T4 trend: {trend:+.1f} pts/year")
