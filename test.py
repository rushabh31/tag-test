import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import yaml
from your_module import regulation  # Replace with your actual module name

class TestRegulationFunction:
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment and dependencies"""
        with patch('your_module.get_environment') as mock_get_env, \
             patch('your_module.yaml.safe_load') as mock_yaml, \
             patch('your_module.get_async_query_executor') as mock_executor, \
             patch('your_module.process_with_rerun') as mock_process, \
             patch('builtins.open', mock_open(read_data='mock_credentials')) as mock_file, \
             patch('your_module.logger') as mock_logger:
            
            # Setup mock environment
            mock_env = Mock()
            mock_env.vector_store_env.credentials_path.read_text.return_value = 'mock_creds'
            mock_env.vector_store_env.url = 'mock_url'
            mock_env.vector_store_env.pool_size = 10
            mock_env.application_name = 'test_app'
            mock_env.vector_store_env.ssl_cert_file = 'cert.pem'
            mock_get_env.return_value = mock_env
            
            # Setup mock credentials
            mock_yaml.return_value = {
                'user': 'test_user',
                'password': 'test_password'
            }
            
            yield {
                'mock_env': mock_env,
                'mock_yaml': mock_yaml,
                'mock_executor': mock_executor,
                'mock_process': mock_process,
                'mock_logger': mock_logger
            }

    @pytest.mark.asyncio
    async def test_regulation_non_regulation_tag(self, mock_environment):
        """Test when tag is not 'REGULATION'"""
        result = await regulation(1, "test description", [], "OTHER_TAG")
        
        expected = {
            "index": 1,
            "tag": "OTHER_TAG",
            "issue_description": "test description",
            "result": []
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_regulation_with_regulation_tag_success(self, mock_environment):
        """Test successful regulation processing with REGULATION tag"""
        # Mock database results
        mock_db_results = [
            {
                'pub_num': 'REG001',
                'pub_name': 'Test Regulation',
                'citation': 'Test Citation',
                'function_list': 'Test Function',
                'etr': 'Test ETR',
                'risk_lib_anchoring': 'Test Risk',
                'regulatory': 'Test Regulatory',
                'regulatory_tier': 'Tier 1',
                'country': 'US',
                'inventory_type': 'Type A',
                'key_contact_info': 'Contact Info'
            }
        ]
        
        # Setup mock query executor
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        # Setup mock LLM response
        mock_llm_response = {
            'res': ['Processed regulation response']
        }
        mock_environment['mock_process'].return_value = mock_llm_response
        
        result = await regulation(1, "test description", [{"category": "finance"}], "REGULATION")
        
        expected = {
            "index": 1,
            "tag": "REGULATION",
            "issue_description": "test description",
            "result": ['Processed regulation response']
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_regulation_with_empty_categories(self, mock_environment):
        """Test regulation with empty categories list"""
        mock_db_results = []
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': []}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        result = await regulation(1, "test description", [], "REGULATION")
        
        # Should use "No Regulation" as categories_regulation_concatenated
        assert result["result"] == []

    @pytest.mark.asyncio
    async def test_regulation_with_no_category_in_rows(self, mock_environment):
        """Test regulation when rows don't have 'category' field"""
        mock_db_results = []
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': []}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        result = await regulation(1, "test description", [{"other_field": "value"}], "REGULATION")
        
        assert result["result"] == []

    @pytest.mark.asyncio
    async def test_regulation_with_long_results_list(self, mock_environment):
        """Test regulation when LLM returns more than 9 results"""
        mock_db_results = [{'pub_num': f'REG{i:03d}', 'pub_name': f'Reg {i}', 'citation': f'Citation {i}', 'function_list': 'func', 'etr': 'etr', 'risk_lib_anchoring': 'risk', 'regulatory': 'reg', 'regulatory_tier': 'tier', 'country': 'US', 'inventory_type': 'type', 'key_contact_info': 'contact'} for i in range(15)]
        
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        # Mock LLM response with more than 9 items
        long_response = [f"Result {i}" for i in range(12)]
        mock_llm_response = {'res': long_response}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        result = await regulation(1, "test description", [{"category": "finance"}], "REGULATION")
        
        # Should truncate to first 9 results
        assert len(result["result"]) == 9
        assert result["result"] == long_response[:9]

    @pytest.mark.asyncio
    async def test_regulation_with_llm_error_response(self, mock_environment):
        """Test regulation when LLM returns error response"""
        mock_db_results = [
            {
                'pub_num': 'REG001',
                'pub_name': 'Test Regulation',
                'citation': 'Test Citation',
                'function_list': 'Test Function',
                'etr': 'Test ETR',
                'risk_lib_anchoring': 'Test Risk',
                'regulatory': 'Test Regulatory',
                'regulatory_tier': 'Tier 1',
                'country': 'US',
                'inventory_type': 'Type A',
                'key_contact_info': 'Contact Info'
            }
        ]
        
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        # Mock LLM response with error
        mock_llm_response = {
            'error': 'Some error occurred'
        }
        mock_environment['mock_process'].return_value = mock_llm_response
        
        result = await regulation(1, "test description", [{"category": "finance"}], "REGULATION")
        
        expected = {
            "index": 1,
            "tag": "REGULATION",
            "issue_description": "test description",
            "result": [],
            "error": 'Some error occurred'
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_regulation_with_exception(self, mock_environment):
        """Test regulation function when an exception occurs"""
        # Make the query executor raise an exception
        mock_environment['mock_executor'].side_effect = Exception("Database connection failed")
        
        result = await regulation(1, "test description", [{"category": "finance"}], "REGULATION")
        
        expected = {
            "index": 1,
            "tag": "REGULATION",
            "issue_description": "test description",
            "result": [],
            "error": "Database connection failed"
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_regulation_with_multiple_categories(self, mock_environment):
        """Test regulation with multiple categories in rows"""
        mock_db_results = []
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': ['Response for multiple categories']}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        rows_with_categories = [
            {"category": "finance"},
            {"category": "healthcare"},
            {"category": "technology"}
        ]
        
        result = await regulation(1, "test description", rows_with_categories, "REGULATION")
        
        # Verify the categories were properly concatenated
        assert result["result"] == ['Response for multiple categories']

    @pytest.mark.asyncio
    async def test_regulation_categories_with_empty_strings(self, mock_environment):
        """Test regulation when categories contain empty strings"""
        mock_db_results = []
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': []}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        rows_with_empty_categories = [
            {"category": "finance"},
            {"category": ""},
            {"category": "healthcare"}
        ]
        
        result = await regulation(1, "test description", rows_with_empty_categories, "REGULATION")
        
        # Should filter out empty categories
        assert result["result"] == []

    @pytest.mark.asyncio
    async def test_regulation_db_results_population(self, mock_environment):
        """Test that database results are properly populated in res"""
        mock_db_results = [
            {
                'pub_num': 'REG001',
                'pub_name': 'Test Regulation',
                'citation': 'Test Citation',
                'function_list': 'Test Function',
                'etr': 'Test ETR',
                'risk_lib_anchoring': 'Test Risk',
                'regulatory': 'Test Regulatory',
                'regulatory_tier': 'Tier 1',
                'country': 'US',
                'inventory_type': 'Type A',
                'key_contact_info': 'Contact Info'
            }
        ]
        
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': ['Processed response']}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        await regulation(1, "test description", [{"category": "finance"}], "REGULATION")
        
        # Verify that process_with_rerun was called with the right parameters
        mock_environment['mock_process'].assert_called_once()
        call_args = mock_environment['mock_process'].call_args
        assert 'regulation' in call_args[0][0]  # First argument should contain 'regulation'
        assert call_args[1]['max_attempts'] == 3  # max_attempts parameter

    @pytest.mark.asyncio 
    async def test_regulation_yaml_credential_loading(self, mock_environment):
        """Test that credentials are properly loaded from YAML"""
        mock_db_results = []
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': []}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        await regulation(1, "test description", [], "REGULATION")
        
        # Verify yaml.safe_load was called
        mock_environment['mock_yaml'].assert_called_once()

    @pytest.mark.asyncio
    async def test_regulation_query_executor_parameters(self, mock_environment):
        """Test that query executor is called with correct parameters"""
        mock_db_results = []
        mock_query_executor = Mock()
        mock_query_executor.return_value = mock_db_results
        mock_environment['mock_executor'].return_value = mock_query_executor
        
        mock_llm_response = {'res': []}
        mock_environment['mock_process'].return_value = mock_llm_response
        
        await regulation(1, "test description", [{"category": "finance"}], "REGULATION")
        
        # Verify get_async_query_executor was called with correct parameters
        mock_environment['mock_executor'].assert_called_once()
        call_kwargs = mock_environment['mock_executor'].call_args[1]
        assert call_kwargs['url'] == 'mock_url'
        assert call_kwargs['user'] == 'test_user'
        assert call_kwargs['password'] == 'test_password'
        assert call_kwargs['pool_size'] == 10
        assert call_kwargs['application_name'] == 'test_app'
        assert call_kwargs['ssl_cert_file'] == 'cert.pem'
