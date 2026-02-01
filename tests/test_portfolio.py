"""
Tests for Portfolio CSV Loading and Validation
"""
import tempfile
import os
import pytest
from app.services.portfolio import load_portfolio_csv, PortfolioValidationResult


class TestPortfolioLoader:
    """Tests for the portfolio CSV loader and validator."""
    
    def test_valid_portfolio(self):
        """Test loading a valid portfolio CSV."""
        content = """symbol,weight
AAPL,0.30
MSFT,0.20
SPY,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is True
            assert len(result.entries) == 3
            assert len(result.errors) == 0
            assert abs(result.total_weight - 1.0) < 0.001
            
            # Check entries
            symbols = {e.symbol for e in result.entries}
            assert symbols == {'AAPL', 'MSFT', 'SPY'}
            
            # Check weights
            weights = {e.symbol: e.weight for e in result.entries}
            assert weights['AAPL'] == 0.30
            assert weights['MSFT'] == 0.20
            assert weights['SPY'] == 0.50
            
            os.unlink(f.name)
    
    def test_duplicate_symbols(self):
        """Test that duplicate symbols are rejected."""
        content = """symbol,weight
AAPL,0.30
AAPL,0.20
SPY,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is False
            assert any('Duplicate symbol' in e for e in result.errors)
            
            os.unlink(f.name)
    
    def test_negative_weights(self):
        """Test that negative weights are rejected."""
        content = """symbol,weight
AAPL,0.30
MSFT,-0.20
SPY,0.90
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is False
            assert any('Negative weight' in e for e in result.errors)
            
            os.unlink(f.name)
    
    def test_weight_sum_not_one(self):
        """Test that weight sum must equal 1.0 within epsilon."""
        content = """symbol,weight
AAPL,0.30
MSFT,0.20
SPY,0.30
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name, epsilon=0.01)
            
            assert result.is_valid is False
            assert any('Weight sum' in e for e in result.errors)
            assert abs(result.total_weight - 0.80) < 0.001
            
            os.unlink(f.name)
    
    def test_weight_sum_within_epsilon(self):
        """Test that weight sum within epsilon is accepted."""
        content = """symbol,weight
AAPL,0.30
MSFT,0.20
SPY,0.505
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            # Sum is 1.005, epsilon of 0.01 should accept
            result = load_portfolio_csv(f.name, epsilon=0.01)
            
            assert result.is_valid is True
            assert len(result.errors) == 0
            
            os.unlink(f.name)
    
    def test_missing_file(self):
        """Test handling of missing file."""
        result = load_portfolio_csv('/nonexistent/path/portfolio.csv')
        
        assert result.is_valid is False
        assert any('not found' in e for e in result.errors)
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        content = """ticker,allocation
AAPL,0.50
MSFT,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is False
            assert any('Missing required columns' in e for e in result.errors)
            
            os.unlink(f.name)
    
    def test_invalid_weight_format(self):
        """Test handling of non-numeric weights."""
        content = """symbol,weight
AAPL,0.30
MSFT,invalid
SPY,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is False
            assert any('Invalid weight' in e for e in result.errors)
            
            os.unlink(f.name)
    
    def test_empty_symbol(self):
        """Test handling of empty symbols."""
        content = """symbol,weight
AAPL,0.30
,0.20
SPY,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is False
            assert any('Empty symbol' in e for e in result.errors)
            
            os.unlink(f.name)
    
    def test_zero_weight_warning(self):
        """Test that zero weights generate warnings but don't fail."""
        content = """symbol,weight
AAPL,0.50
MSFT,0
SPY,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is True
            assert any('Zero weight' in w for w in result.warnings)
            
            os.unlink(f.name)
    
    def test_symbol_case_normalization(self):
        """Test that symbols are normalized to uppercase."""
        content = """symbol,weight
aapl,0.30
Msft,0.20
SPY,0.50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = load_portfolio_csv(f.name)
            
            assert result.is_valid is True
            symbols = {e.symbol for e in result.entries}
            assert symbols == {'AAPL', 'MSFT', 'SPY'}
            
            os.unlink(f.name)
    
    def test_real_portfolio_file(self):
        """Test loading the actual target portfolio file."""
        # This test validates the real data file in the repository
        import os
        
        # Get the path relative to tests
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        portfolio_path = os.path.join(base_dir, 'data', 'target_portfolio.csv')
        
        if os.path.exists(portfolio_path):
            result = load_portfolio_csv(portfolio_path)
            
            # The real file should be valid
            assert result.is_valid is True, f"Errors: {result.errors}"
            assert len(result.entries) > 0
            assert len(result.errors) == 0
            
            # Weight sum should be ~1.0
            assert abs(result.total_weight - 1.0) < 0.01
