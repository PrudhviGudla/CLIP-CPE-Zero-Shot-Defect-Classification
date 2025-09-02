"""
Spec containing the configuration for the defect classification task
"""
from pydantic import BaseModel
from typing import List, Dict

class DefectClassificationSpec(BaseModel):
    """
    Configuration for defect classification.

    Attributes:
        model_name (str): Name of the CLIP model variant to use.
        normal_states (List[str]): List of state words for normal images.
        anomalous_states (List[str]): List of state words for anomalous images.
        defect_states (Dict[str, List[str]]): Dictionary of defect types and their corresponding state words.
        templates (List[str]): Prompt templates for CPE
        temperature (float): Temperature parameter for the CLIP model
    """

    # CLIP model configuration
    model_name: str = 'ViT-L/14'

    normal_states: List[str] = [
    # Quality descriptors
    "perfect", "flawless", "pristine", "intact", "unblemished",
    "high quality", "acceptable", "standard", "normal", "healthy", "good",

    # Shape and form descriptors
    "well formed", "properly shaped", "regular", "uniform", "symmetrical",

    # Surface and appearance descriptors
    "smooth", "clean", "clear", "bright", "fresh", "pure",

    # Structural integrity descriptors
    "solid", "whole", "complete", "unbroken", "sound", "robust", "firm",

    # Commercial quality descriptors
    "premium", "grade A", "first class", "superior", "excellent", "fine",

    # Condition descriptors
    "undamaged", "unmarked", "unprocessed", "natural", "untouched"
    # ,
    # "original", "best", "better"
]

    anomalous_states: List[str] = [
        "defective", "damaged", "broken", "faulty", "rejected",
        "poor quality", "unacceptable", "failed", "bad", "abnormal", "with flaw", "with defect", "with damage"
        # ,
        # "print damage", "cut damage", "crack damage", "hole damage"
    ]

    # Multi-class defect classification states
    defect_states: Dict[str, List[str]] = {
    'crack': [
        # Basic crack descriptors
        'cracked', 'fractured', 'split', 'broken',

        # Crack-specific terminology
        'fissured', 'creviced', 'cleft', 'rent', 'ruptured',
        'showing crack lines', 'with linear fractures', 'split open',
        'stress cracked', 'surface cracked', 'shell cracked',

        # Crack severity descriptors
        'hairline cracked', 'deeply cracked', 'partially cracked',
        'showing crack damage', 'with visible cracks'
    ],

    'cut': [
        # Basic cut descriptors
        'cut', 'sliced', 'severed', 'chopped',

        # Cut-specific terminology
        'incised', 'carved', 'gouged', 'scored', 'nicked',
        'with clean cuts', 'showing knife marks', 'mechanically cut',
        'sharp edge damaged', 'blade damaged', 'slicing damaged',

        # Cut pattern descriptors
        'straight cut', 'diagonal cut', 'partially cut',
        'showing cutting damage', 'with cut marks'
    ],

    'hole': [
        # Basic hole descriptors
        'holed', 'punctured', 'perforated', 'pierced',

        # Hole-specific terminology
        'bored', 'drilled', 'punched', 'penetrated', 'breached',
        'with round holes', 'showing void damage', 'cavity damaged',
        'tunnel damaged', 'bore hole damaged', 'entry hole damaged',

        # Hole size and type descriptors
        'pinhole damaged', 'large hole damaged', 'small hole damaged',
        'showing hole defect', 'with circular voids', 'hollow damaged'
    ],

    'print': [
        # Basic print descriptors
        'printed', 'marked', 'stained', 'discolored',

        # Print-specific terminology (from research findings)
        'ink stained', 'color marked', 'pigment stained', 'dye marked',
        'with white markings', 'showing print impressions', 'transfer printed',
        'contamination marked', 'residue stained', 'coating marked',

        # Print pattern descriptors
        'spotted with ink', 'streaked', 'blotched', 'smudged',
        'showing print damage', 'with surface contamination',
        'artificially marked', 'processing stained', 'white print markings'
    ]
}

    # CPE templates
    templates: List[str] = [
    'a cropped photo of the {} hazelnut',
    'a cropped photo of a {} hazelnut',
    'a close-up photo of a {} hazelnut',
    'a close-up photo of the {} hazelnut',
    'a bright photo of a {} hazelnut',
    'a bright photo of the {} hazelnut',
    'a dark photo of the {} hazelnut',
    'a dark photo of a {} hazelnut',
    'a jpeg corrupted photo of a {} hazelnut',
    'a jpeg corrupted photo of the {} hazelnut',
    'a blurry photo of the {} hazelnut',
    'a blurry photo of a {} hazelnut',
    'a photo of a {} hazelnut',
    'a photo of the {} hazelnut',
    'a photo of a small {} hazelnut',
    'a photo of the small {} hazelnut',
    'a photo of a large {} hazelnut',
    'a photo of the large {} hazelnut',
    'a photo of the {} hazelnut for visual inspection',
    'a photo of a {} hazelnut for visual inspection',
    'a photo of the {} hazelnut for anomaly detection',
    'a photo of a {} hazelnut for anomaly detection'
    ]

    # WinCLIP parameters
    temperature: float = 0.2
