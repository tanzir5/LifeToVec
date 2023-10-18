import pandas as pd

from dataclasses import dataclass, field
from src.data_new.vocabulary import Vocabulary
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, cast

import dask
import pandas as pd

from src.data_new.decorators import save_pickle, save_tsv
from src.data_new.serialize import DATA_ROOT
from src.data_new.sources.base import TokenSource

from src.new_code.constants import BIRTH_YEAR, BIRTH_MONTH, ORIGIN, GENDER, DELIMITER, TIME_COLUMNS, PRIMARY_KEY, IGNORE_COLUMNS

class DataFile():
  
  def __init__(self, path, name=None):
    self.path = path
    self.name = name
    self.df = None

  def _get_df(self):
    if self.df is None:
      self.df = pd.read_csv(self.path, delimiter=DELIMITER)
    return self.df

  def _get_unique_tokens_for_column(self, data, column):
    tokens_with_column_name = [f"{column}_{str(d)}" for d in data.unique()]

    return pd.DataFrame({
        "TOKEN": tokens_with_column_name,
        "CATEGORY": f"{self.name}_{column}",
      }
    )

  def get_all_unique_tokens_with_category(self):
    df = self._get_df()
    unique_tokens_by_category = []
    for column in df.columns:
      if (
        column not in TIME_COLUMNS and 
        column not in IGNORE_COLUMNS and 
        column != PRIMARY_KEY
      ):
        unique_tokens_by_category.append(
          self._get_unique_tokens_for_column(
            df[column], column,
          )
        )
    return unique_tokens_by_category

@dataclass
class CustomVocabulary(Vocabulary):
    """
    Generate a vocabulary from the tokenized training data of a corpus.

    :param corpus: The :class:`src.data_new.Corpus` to generate the vocabulary from.
    :param name: Name of the vocabulary.
    :param general_tokens: General tokens.
    :param background_tokens: Background tokens.
    :param year_range: Range of years (inclusive) to generate tokens for.
    :param min_token_count: The minimum number of occurances of a token to be included
        in the vocabulary.
    :param min_token_count_field: Field-specific minimum token counts.

    """
    
    name: str
    data_files: List[DataFile]
    general_tokens: List[str] = field(
        default_factory=lambda: [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[UNK]",
        ]
    )
    background_tokens: List[str] = field(
        default_factory=lambda: [f"{GENDER}_1", f"{GENDER}_2",]
    )
    year_range: Tuple[int, int] = field(
        default_factory=lambda: (1, 138)  # inclusive
    )
    origin_range: Tuple[int, int] = field(
        default_factory=lambda: (1, 200) # inclusive
    )
    min_token_count: int = 1000 
    min_token_count_field: Dict[str, int] = field(default_factory=dict)

    vocab_df = None

    # @save_tsv(DATA_ROOT / "processed/vocab/{self.name}/", on_validation_error="error")
    def vocab(self) -> pd.DataFrame:
        """Filters the tokens by count, sorts them lexicographically for each source,
        and computes the voculary with the field labels as categories.
        """
        if self.vocab_df is not None:
          return self.vocab_df

        general = pd.DataFrame({"TOKEN": self.general_tokens, "CATEGORY": "GENERAL"})
        background = pd.DataFrame(
            {"TOKEN": self.background_tokens, "CATEGORY": "BACKGROUND"}
        )
        month = pd.DataFrame(
            {"TOKEN": [f"{BIRTH_MONTH}_{i}" for i in range(1, 13)], "CATEGORY": BIRTH_MONTH}
        )
        year = pd.DataFrame(
            {
                "TOKEN": [
                    f"{BIRTH_YEAR}_{i}"
                    for i in range(self.year_range[0], self.year_range[1] + 1)
                ],
                "CATEGORY": BIRTH_YEAR,
            }
        )
        origin = pd.DataFrame(
            {
                "TOKEN": [
                    f"{ORIGIN}_{i}"
                    for i in range(self.origin_range[0], self.origin_range[1] + 1)
                ],
                "CATEGORY": ORIGIN,
            }
        )
        
        vocab_parts = [general, background, month, year, origin]
        for source_file in self.data_files:
          vocab_parts.extend(
            source_file.get_all_unique_tokens_with_category()
          )
        self.vocab_df = pd.concat(
          vocab_parts, ignore_index=True
        ).rename_axis(index="ID")
        self.vocab_df['ID'] = self.vocab_df.index

        return self.vocab_df  