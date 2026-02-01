"""
Portfolio CSV Loader and Validator

Loads and validates the target portfolio from a CSV file.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class PortfolioEntry:
    """A single entry in the target portfolio."""
    symbol: str
    weight: float


@dataclass
class PortfolioValidationResult:
    """Result of portfolio validation."""
    is_valid: bool
    entries: List[PortfolioEntry]
    errors: List[str]
    warnings: List[str]
    total_weight: float


def load_portfolio_csv(file_path: str, epsilon: float = 0.01) -> PortfolioValidationResult:
    """
    Load and validate a target portfolio from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        epsilon: Tolerance for weight sum validation (default 0.01)
    
    Returns:
        PortfolioValidationResult with validation status and parsed entries
    
    The CSV must have columns: symbol, weight
    Validation checks:
    - No duplicate symbols
    - No negative weights
    - Sum of weights ≈ 1.0 (within epsilon)
    """
    errors: List[str] = []
    warnings: List[str] = []
    entries: List[PortfolioEntry] = []
    
    path = Path(file_path)
    
    # Check file exists
    if not path.exists():
        return PortfolioValidationResult(
            is_valid=False,
            entries=[],
            errors=[f"Portfolio file not found: {file_path}"],
            warnings=[],
            total_weight=0.0
        )
    
    # Check file is readable
    if not path.is_file():
        return PortfolioValidationResult(
            is_valid=False,
            entries=[],
            errors=[f"Path is not a file: {file_path}"],
            warnings=[],
            total_weight=0.0
        )
    
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            if reader.fieldnames is None:
                return PortfolioValidationResult(
                    is_valid=False,
                    entries=[],
                    errors=["CSV file is empty or has no headers"],
                    warnings=[],
                    total_weight=0.0
                )
            
            required_columns = {'symbol', 'weight'}
            actual_columns = set(reader.fieldnames)
            missing_columns = required_columns - actual_columns
            
            if missing_columns:
                return PortfolioValidationResult(
                    is_valid=False,
                    entries=[],
                    errors=[f"Missing required columns: {', '.join(missing_columns)}"],
                    warnings=[],
                    total_weight=0.0
                )
            
            # Track seen symbols for duplicate detection
            seen_symbols: set = set()
            line_num = 1  # Header is line 1
            
            for row in reader:
                line_num += 1
                symbol = row.get('symbol', '').strip().upper()
                weight_str = row.get('weight', '').strip()
                
                # Validate symbol
                if not symbol:
                    errors.append(f"Line {line_num}: Empty symbol")
                    continue
                
                # Check for duplicate symbols
                if symbol in seen_symbols:
                    errors.append(f"Line {line_num}: Duplicate symbol '{symbol}'")
                    continue
                seen_symbols.add(symbol)
                
                # Validate weight
                try:
                    weight = float(weight_str)
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid weight '{weight_str}' for symbol '{symbol}'")
                    continue
                
                # Check for negative weights
                if weight < 0:
                    errors.append(f"Line {line_num}: Negative weight {weight} for symbol '{symbol}'")
                    continue
                
                # Check for zero weights (warning, not error)
                if weight == 0:
                    warnings.append(f"Line {line_num}: Zero weight for symbol '{symbol}'")
                
                # Check for weights > 1 (warning, could be intentional for leveraged)
                if weight > 1:
                    warnings.append(f"Line {line_num}: Weight {weight} > 1.0 for symbol '{symbol}'")
                
                entries.append(PortfolioEntry(symbol=symbol, weight=weight))
    
    except csv.Error as e:
        return PortfolioValidationResult(
            is_valid=False,
            entries=[],
            errors=[f"CSV parsing error: {str(e)}"],
            warnings=[],
            total_weight=0.0
        )
    except Exception as e:
        return PortfolioValidationResult(
            is_valid=False,
            entries=[],
            errors=[f"Error reading file: {str(e)}"],
            warnings=[],
            total_weight=0.0
        )
    
    # Check we have at least one entry
    if not entries:
        errors.append("No valid portfolio entries found")
    
    # Calculate total weight
    total_weight = sum(entry.weight for entry in entries)
    
    # Validate weight sum
    if entries and abs(total_weight - 1.0) > epsilon:
        errors.append(
            f"Weight sum is {total_weight:.6f}, expected 1.0 (tolerance: {epsilon})"
        )
    
    is_valid = len(errors) == 0
    
    return PortfolioValidationResult(
        is_valid=is_valid,
        entries=entries,
        errors=errors,
        warnings=warnings,
        total_weight=total_weight
    )


# Cached portfolio to avoid re-reading file on every request
_cached_portfolio: Optional[PortfolioValidationResult] = None
_cached_file_path: Optional[str] = None


def get_portfolio(file_path: str, epsilon: float = 0.01, force_reload: bool = False) -> PortfolioValidationResult:
    """
    Get the portfolio, using cache if available.
    
    Args:
        file_path: Path to the CSV file
        epsilon: Tolerance for weight sum validation
        force_reload: Force reload from disk
    
    Returns:
        PortfolioValidationResult
    """
    global _cached_portfolio, _cached_file_path
    
    if force_reload or _cached_portfolio is None or _cached_file_path != file_path:
        _cached_portfolio = load_portfolio_csv(file_path, epsilon)
        _cached_file_path = file_path
    
    return _cached_portfolio


def clear_portfolio_cache():
    """Clear the portfolio cache."""
    global _cached_portfolio, _cached_file_path
    _cached_portfolio = None
    _cached_file_path = None
