import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import asyncio

# Mock data - completely isolated from real systems
MOCK_REQUIREMENT_DATA = [
    {
        "reqId": "REQ001",
        "reqName": "Sample Requirement",
        "requirementSummaryEnglishText": "<p>This is a <b>test</b> requirement</p>",
        "citation": "Test Citation 2024",
        "functionList": [{"name": "function1"}, {"name": "function2"}],
        "etr": "test_etr_value",
        "riskLibAnchoring": {"level": "high", "score": 9},
        "regulatory": "Test Regulatory Body",
        "regulatoryTier": "Tier 1",
        "regulatoryCountry": "US",
        "inventoryType": "compliance",
        "keyContactInfo": {"email": "test@example.com", "phone": "555-0123"}
    },
    {
        "reqId": "REQ002",
        "reqName": "Another Requirement",
        "requirementSummaryEnglishText": "<div>HTML content here</div>",
        "citation": "Another Citation",
        "functionList": [],
        "etr": None,
        "riskLibAnchoring": {},
        "regulatory": "",
        "regulatoryTier": "Tier 2",
        "regulatoryCountry": "CA",
        "inventoryType": "operational",
        "keyContactInfo": {}
    }
]

MOCK_COUNTRIES_DATA = [
    {"country": "US"},
    {"country": "CA"},
    {"country": "UK"}
]

MOCK_CREDENTIALS = {
    "user": "test_user",
    "password": "test_password"
}

class TestSyncAttributesToDatabase:
    
    @pytest.mark.asyncio
    async def test_sync_attributes_success(self):
        """Test successful sync - all mocked"""
        mock_results = [
            ["Activity taxonomy synced"],
            ["Control taxonomy synced"],
            ["Risk taxonomy synced"],
            ["Root cause taxonomy synced"],
            ["Regulation inventory synced"]
        ]
        
        with patch('your_module.sync_activity_taxonomy', return_value=mock_results[0]), \
             patch('your_module.sync_control_taxonomy', return_value=mock_results[1]), \
             patch('your_module.sync_risk_taxonomy', return_value=mock_results[2]), \
             patch('your_module.sync_root_cause_taxonomy', return_value=mock_results[3]), \
             patch('your_module.sync_reg_inventory', return_value=mock_results[4]), \
             patch('your_module.get_async_transaction_executor', return_value=AsyncMock()), \
             patch('your_module.get_async_query_executor', return_value=AsyncMock()), \
             patch('your_module.get_environment', return_value=MagicMock()), \
             patch('your_module.GrcServiceSettings', return_value=MagicMock()):
            
            # Import and test your actual function
            from your_module import sync_attributes_to_database
            
            result = await sync_attributes_to_database()
            
            expected = "Activity taxonomy synced|Control taxonomy synced|Risk taxonomy synced|Root cause taxonomy synced|Regulation inventory synced"
            assert result == expected

    @pytest.mark.asyncio
    async def test_sync_attributes_with_exception(self):
        """Test when one sync function fails"""
        with patch('your_module.sync_activity_taxonomy', side_effect=Exception("Activity sync failed")), \
             patch('your_module.sync_control_taxonomy', return_value=["Control synced"]), \
             patch('your_module.sync_risk_taxonomy', return_value=["Risk synced"]), \
             patch('your_module.sync_root_cause_taxonomy', return_value=["Root cause synced"]), \
             patch('your_module.sync_reg_inventory', return_value=["Regulation synced"]), \
             patch('your_module.get_async_transaction_executor', return_value=AsyncMock()), \
             patch('your_module.get_async_query_executor', return_value=AsyncMock()), \
             patch('your_module.get_environment', return_value=MagicMock()), \
             patch('your_module.GrcServiceSettings', return_value=MagicMock()):
            
            from your_module import sync_attributes_to_database
            
            with pytest.raises(Exception, match="Activity sync failed"):
                await sync_attributes_to_database()

class TestSyncRegInventory:
    
    @pytest.fixture
    def mock_transaction_executor(self):
        """Create mock transaction executor"""
        mock_executor = AsyncMock()
        mock_executor.return_value = MagicMock()
        return mock_executor
    
    @pytest.fixture
    def mock_query_executor(self):
        """Create mock query executor"""
        mock_executor = AsyncMock()
        mock_executor.return_value = MOCK_COUNTRIES_DATA
        return mock_executor
    
    @pytest.fixture
    def mock_grc_config(self):
        """Create mock GRC service config"""
        config = MagicMock()
        config.api_url = "https://mock-api.example.com"
        return config
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment"""
        env = MagicMock()
        env.vector_store_env.pool_size = 5
        env.application_name = "test_app"
        env.vector_store_env.ssl_cert_file = "/path/to/cert.pem"
        env.vector_store_env.url = "postgresql://localhost:5432/test"
        env.vector_store_env.credentials_path.read_text.return_value = json.dumps(MOCK_CREDENTIALS)
        return env
    
    @pytest.fixture
    def mock_api_response(self):
        """Create mock API response"""
        response = MagicMock()
        response.json.return_value = {
            "data": {
                "requirementSelectionData": MOCK_REQUIREMENT_DATA
            }
        }
        return response

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_success(self, mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment, mock_api_response):
        """Test successful regulation inventory sync"""
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.GrcServiceSettings', return_value=mock_grc_config), \
             patch('your_module.RegServiceSettings', return_value=MagicMock()), \
             patch('your_module.get_async_transaction_executor', return_value=mock_transaction_executor), \
             patch('your_module.get_async_query_executor', return_value=mock_query_executor), \
             patch('your_module.requests.post', return_value=mock_api_response), \
             patch('your_module.truncate_db', return_value="Table truncated"), \
             patch('your_module.remove_html_tags', return_value="Clean text"), \
             patch('your_module.json.dumps', return_value='{"mock": "data"}'), \
             patch('your_module.logger') as mock_logger:
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment)
            
            # Verify result contains expected messages
            assert isinstance(result, list)
            assert len(result) > 0
            assert "Starting sync_reg_inventory" in result[0]
            
            # Verify mocks were called
            mock_query_executor.assert_called()
            mock_transaction_executor.assert_called()

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_no_countries(self, mock_transaction_executor, mock_grc_config, mock_environment):
        """Test when no countries are returned from database"""
        
        # Mock query executor to return empty list
        empty_query_executor = AsyncMock()
        empty_query_executor.return_value = []
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.get_async_query_executor', return_value=empty_query_executor):
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(mock_transaction_executor, empty_query_executor, 
                                            mock_grc_config, mock_environment)
            
            assert "No records found in EIM.REGULATION_T1_COUNTRY table" in str(result)

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_api_failure(self, mock_transaction_executor, mock_query_executor, 
                                                mock_grc_config, mock_environment):
        """Test API request failure"""
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.requests.post', side_effect=Exception("API connection failed")):
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment)
            
            assert "Error fetching records from" in str(result)

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_empty_api_response(self, mock_transaction_executor, mock_query_executor, 
                                                       mock_grc_config, mock_environment):
        """Test when API returns empty data"""
        
        empty_response = MagicMock()
        empty_response.json.return_value = {
            "data": {
                "requirementSelectionData": []
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.requests.post', return_value=empty_response):
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment)
            
            assert "api has not returned any data in requirement_data" in str(result)

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_database_insert_failure(self, mock_query_executor, 
                                                            mock_grc_config, mock_environment, mock_api_response):
        """Test database insertion failure"""
        
        # Mock transaction executor to fail
        failing_executor = AsyncMock()
        failing_executor.side_effect = Exception("Database insert failed")
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.requests.post', return_value=mock_api_response), \
             patch('your_module.truncate_db', return_value="Truncated"), \
             patch('your_module.remove_html_tags', return_value="Clean text"):
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(failing_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment)
            
            assert "Error inserting records into" in str(result)

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_truncate_failure(self, mock_transaction_executor, mock_query_executor, 
                                                     mock_grc_config, mock_environment):
        """Test table truncation failure"""
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.truncate_db', side_effect=Exception("Truncate failed")):
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment)
            
            assert "Error truncating" in str(result)

class TestFlattenRiskRecords:
    
    @pytest.mark.asyncio
    async def test_flatten_risk_records_success(self):
        """Test successful risk records flattening"""
        mock_transaction_executor = AsyncMock()
        mock_query_executor = AsyncMock()
        
        with patch('your_module.logger') as mock_logger:
            from your_module import flatten_risk_records
            
            await flatten_risk_records(mock_transaction_executor, mock_query_executor)
            
            # Verify SQL was executed
            mock_transaction_executor.assert_called_once()
            mock_logger.info.assert_called_with("complete flatten_risk_records")

    @pytest.mark.asyncio
    async def test_flatten_risk_records_failure(self):
        """Test risk records flattening failure"""
        mock_transaction_executor = AsyncMock()
        mock_query_executor = AsyncMock()
        
        # Make the transaction executor fail
        mock_transaction_executor.side_effect = Exception("SQL execution failed")
        
        with patch('your_module.logger') as mock_logger:
            from your_module import flatten_risk_records
            
            await flatten_risk_records(mock_transaction_executor, mock_query_executor)
            
            mock_logger.info.assert_called_with("Error flattening records: SQL execution failed")

class TestRemoveHtmlTags:
    
    def test_remove_html_tags_success(self):
        """Test successful HTML tag removal"""
        html_input = "<p>This is <b>bold</b> text with <i>italics</i></p>"
        expected_output = "This is bold text with italics"
        
        # Mock BeautifulSoup
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = expected_output
            
            from your_module import remove_html_tags
            
            result = remove_html_tags(html_input)
            
            assert result == expected_output
            mock_soup.assert_called_once_with(html_input, "html.parser")

    def test_remove_html_tags_beautifulsoup_failure(self):
        """Test HTML tag removal when BeautifulSoup fails"""
        html_input = "<p>Test content</p>"
        fallback_output = "Test content"
        
        with patch('your_module.BeautifulSoup', side_effect=Exception("BeautifulSoup failed")), \
             patch('your_module.re.sub', return_value=fallback_output), \
             patch('your_module.logger') as mock_logger:
            
            from your_module import remove_html_tags
            
            result = remove_html_tags(html_input)
            
            assert result == fallback_output
            mock_logger.info.assert_called_with("Error parsing HTML with BeautifulSoup: BeautifulSoup failed")

    def test_remove_html_tags_empty_input(self):
        """Test with empty input"""
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = ""
            
            from your_module import remove_html_tags
            
            result = remove_html_tags("")
            
            assert result == ""

    def test_remove_html_tags_none_input(self):
        """Test with None input"""
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = ""
            
            from your_module import remove_html_tags
            
            result = remove_html_tags(None)
            
            assert result == ""

class TestEdgeCases:
    
    @pytest.mark.asyncio
    async def test_sync_with_malformed_requirement_data(self, mock_transaction_executor, mock_query_executor, 
                                                       mock_grc_config, mock_environment):
        """Test with malformed requirement data"""
        
        malformed_data = [
            {
                "reqId": None,
                "reqName": 12345,  # Number instead of string
                "requirementSummaryEnglishText": None,
                "citation": "",
                "functionList": "not_a_list",  # String instead of list
                "etr": {},  # Dict instead of string
                "riskLibAnchoring": "not_a_dict",  # String instead of dict
                "regulatory": None,
                "regulatoryTier": None,
                "regulatoryCountry": None,
                "inventoryType": None,
                "keyContactInfo": "not_a_dict"  # String instead of dict
            }
        ]
        
        malformed_response = MagicMock()
        malformed_response.json.return_value = {
            "data": {
                "requirementSelectionData": malformed_data
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=MOCK_CREDENTIALS), \
             patch('your_module.requests.post', return_value=malformed_response), \
             patch('your_module.truncate_db', return_value="Truncated"), \
             patch('your_module.remove_html_tags', return_value=""), \
             patch('your_module.json.dumps', return_value='""'):
            
            from your_module import sync_reg_inventory
            
            result = await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config, mock_environment)
            
            # Should handle malformed data gracefully
            assert isinstance(result, list)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_credentials_file_not_found(self, mock_transaction_executor, mock_query_executor, 
                                            mock_grc_config):
        """Test when credentials file is not found"""
        
        mock_env = MagicMock()
        mock_env.vector_store_env.credentials_path.read_text.side_effect = FileNotFoundError("File not found")
        
        with patch('your_module.get_environment', return_value=mock_env):
            from your_module import sync_reg_inventory
            
            with pytest.raises(FileNotFoundError):
                await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                        mock_grc_config, mock_env)

    @pytest.mark.asyncio
    async def test_invalid_json_in_credentials(self, mock_transaction_executor, mock_query_executor, 
                                             mock_grc_config, mock_environment):
        """Test when credentials file contains invalid JSON"""
        
        mock_environment.vector_store_env.credentials_path.read_text.return_value = "invalid json content"
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', side_effect=Exception("Invalid YAML")):
            
            from your_module import sync_reg_inventory
            
            with pytest.raises(Exception, match="Invalid YAML"):
                await sync_reg_inventory(mock_transaction_executor, mock_query_executor, 
                                        mock_grc_config, mock_environment)

# Simple test runner to avoid hanging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
