import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import json
import yaml
from your_module import (
    sync_attributes_to_database,
    sync_reg_inventory,
    flatten_risk_records,
    remove_html_tags
)

class TestSyncAttributesToDatabase:
    
    @pytest.mark.asyncio
    async def test_sync_attributes_success(self):
        """Test successful sync of all attributes"""
        with patch('your_module.sync_activity_taxonomy', return_value=["Activity complete"]) as mock_activity, \
             patch('your_module.sync_control_taxonomy', return_value=["Control complete"]) as mock_control, \
             patch('your_module.sync_risk_taxonomy', return_value=["Risk complete"]) as mock_risk, \
             patch('your_module.sync_root_cause_taxonomy', return_value=["Root cause complete"]) as mock_root, \
             patch('your_module.sync_reg_inventory', return_value=["Regulation complete"]) as mock_reg, \
             patch('your_module.get_async_transaction_executor') as mock_trans, \
             patch('your_module.get_async_query_executor') as mock_query, \
             patch('your_module.get_environment') as mock_env, \
             patch('your_module.GrcServiceSettings') as mock_grc:
            
            result = await sync_attributes_to_database()
            
            assert result == "Activity complete|Control complete|Risk complete|Root cause complete|Regulation complete"
            mock_activity.assert_called_once()
            mock_control.assert_called_once()
            mock_risk.assert_called_once()
            mock_root.assert_called_once()
            mock_reg.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_attributes_individual_failures(self):
        """Test when individual sync functions fail"""
        with patch('your_module.sync_activity_taxonomy', side_effect=Exception("Activity failed")) as mock_activity, \
             patch('your_module.sync_control_taxonomy', return_value=["Control complete"]) as mock_control, \
             patch('your_module.sync_risk_taxonomy', return_value=["Risk complete"]) as mock_risk, \
             patch('your_module.sync_root_cause_taxonomy', return_value=["Root cause complete"]) as mock_root, \
             patch('your_module.sync_reg_inventory', return_value=["Regulation complete"]) as mock_reg, \
             patch('your_module.get_async_transaction_executor'), \
             patch('your_module.get_async_query_executor'), \
             patch('your_module.get_environment'), \
             patch('your_module.GrcServiceSettings'):
            
            # This should test if the function handles individual failures gracefully
            with pytest.raises(Exception, match="Activity failed"):
                await sync_attributes_to_database()

class TestSyncRegInventory:
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment with all required attributes"""
        env = MagicMock()
        env.vector_store_env.pool_size = 10
        env.application_name = "test_app"
        env.vector_store_env.ssl_cert_file = "test.cert"
        env.vector_store_env.url = "postgresql://test"
        env.vector_store_env.credentials_path = MagicMock()
        env.vector_store_env.credentials_path.read_text.return_value = '{"user": "test", "password": "test"}'
        return env

    @pytest.fixture
    def mock_credentials(self):
        return {"user": "testuser", "password": "testpass"}

    @pytest.fixture
    def sample_requirement_data(self):
        return [
            {
                "reqId": "REQ001",
                "reqName": "Test Requirement",
                "requirementSummaryEnglishText": "<p>Test <b>summary</b></p>",
                "citation": "Test citation",
                "functionList": [{"name": "func1"}, {"name": "func2"}],
                "etr": "test_etr",
                "riskLibAnchoring": {"level": "high", "score": 8},
                "regulatory": "test_regulatory",
                "regulatoryTier": "Tier 1",
                "regulatoryCountry": "US",
                "inventoryType": "test_type",
                "keyContactInfo": {"contact": "test@example.com", "phone": "123-456-7890"}
            }
        ]

    @pytest.mark.asyncio
    async def test_sync_reg_inventory_full_success(self, mock_environment, mock_credentials, sample_requirement_data):
        """Test complete successful sync with all steps"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        grc_service_config.api_url = "https://test-api.com"
        
        # Mock successful database operations
        transaction_executor.return_value = MagicMock()
        query_executor.return_value = [{"country": "US"}, {"country": "CA"}]
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "requirementSelectionData": sample_requirement_data
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.GrcServiceSettings', return_value=grc_service_config), \
             patch('your_module.RegServiceSettings', return_value=MagicMock()), \
             patch('your_module.get_async_transaction_executor', return_value=transaction_executor), \
             patch('your_module.get_async_query_executor', return_value=query_executor), \
             patch('your_module.requests.post', return_value=mock_response), \
             patch('your_module.truncate_db', return_value="Table truncated successfully"), \
             patch('your_module.remove_html_tags', return_value="Test summary"), \
             patch('your_module.json.dumps', return_value='{"test": "data"}'), \
             patch('your_module.logger') as mock_logger:
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            # Verify all major steps were executed
            assert "Starting sync_reg_inventory" in result[0]
            assert "Number of regulations fetched from API" in str(result)
            mock_logger.info.assert_any_call("Going to insert 1 records in eim.regulation_t1_new table")

    @pytest.mark.asyncio
    async def test_credentials_loading_failure(self, mock_environment):
        """Test credentials loading failure"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        mock_environment.vector_store_env.credentials_path.read_text.side_effect = FileNotFoundError("Credentials file not found")
        
        with patch('your_module.get_environment', return_value=mock_environment):
            with pytest.raises(FileNotFoundError):
                await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)

    @pytest.mark.asyncio
    async def test_yaml_parsing_failure(self, mock_environment):
        """Test YAML parsing failure"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        mock_environment.vector_store_env.credentials_path.read_text.return_value = "invalid: yaml: content:"
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")):
            with pytest.raises(yaml.YAMLError):
                await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)

    @pytest.mark.asyncio
    async def test_countries_query_failure(self, mock_environment, mock_credentials):
        """Test failure when querying countries"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        query_executor.side_effect = Exception("Database connection failed")
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "Error fetching countries from EIM.REGULATION_T1_COUNTRY table" in str(result)

    @pytest.mark.asyncio
    async def test_api_request_failure(self, mock_environment, mock_credentials):
        """Test API request failure"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        query_executor.return_value = [{"country": "US"}]
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.requests.post', side_effect=Exception("API request failed")):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "Error fetching records from" in str(result)

    @pytest.mark.asyncio
    async def test_json_parsing_failure(self, mock_environment, mock_credentials):
        """Test JSON response parsing failure"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        query_executor.return_value = [{"country": "US"}]
        
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.requests.post', return_value=mock_response):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "Error fetching records from" in str(result)

    @pytest.mark.asyncio
    async def test_empty_api_response(self, mock_environment, mock_credentials):
        """Test when API returns empty data"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        query_executor.return_value = [{"country": "US"}]
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "requirementSelectionData": []
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.requests.post', return_value=mock_response):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "api has not returned any data in requirement_data" in str(result)

    @pytest.mark.asyncio
    async def test_truncate_table_failure(self, mock_environment, mock_credentials):
        """Test table truncation failure"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        query_executor.return_value = [{"country": "US"}]
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.truncate_db', side_effect=Exception("Truncate failed")):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "Error truncating eim.regulation_t1_new table" in str(result)

    @pytest.mark.asyncio
    async def test_database_insertion_failure(self, mock_environment, mock_credentials, sample_requirement_data):
        """Test database insertion failure"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        query_executor.return_value = [{"country": "US"}]
        transaction_executor.side_effect = Exception("Insert failed")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "requirementSelectionData": sample_requirement_data
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.requests.post', return_value=mock_response), \
             patch('your_module.truncate_db', return_value="Truncated"), \
             patch('your_module.remove_html_tags', return_value="Test summary"):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "Error inserting records into eim.regulation_t1_new table" in str(result)

    @pytest.mark.asyncio
    async def test_different_field_data_types(self, mock_environment, mock_credentials):
        """Test with different data types in requirement fields"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        # Mock data with various field types
        requirement_data = [
            {
                "reqId": None,  # None value
                "reqName": 123,  # Integer instead of string
                "requirementSummaryEnglishText": "",  # Empty string
                "citation": None,
                "functionList": None,  # None instead of list
                "etr": "",
                "riskLibAnchoring": None,  # None instead of dict
                "regulatory": "",
                "regulatoryTier": None,
                "regulatoryCountry": "",
                "inventoryType": "",
                "keyContactInfo": None  # None instead of dict
            }
        ]
        
        query_executor.return_value = [{"country": "US"}]
        transaction_executor.return_value = MagicMock()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "requirementSelectionData": requirement_data
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.requests.post', return_value=mock_response), \
             patch('your_module.truncate_db', return_value="Truncated"), \
             patch('your_module.remove_html_tags', return_value=""), \
             patch('your_module.json.dumps', return_value='""'):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "Number of regulations fetched from API" in str(result)

class TestFlattenRiskRecords:
    
    @pytest.mark.asyncio
    async def test_flatten_risk_records_success(self):
        """Test successful risk records flattening"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        
        with patch('your_module.logger') as mock_logger:
            await flatten_risk_records(transaction_executor, query_executor)
            
            # Verify SQL execution
            transaction_executor.assert_called_once()
            mock_logger.info.assert_called_with("complete flatten_risk_records")

    @pytest.mark.asyncio
    async def test_flatten_risk_records_sql_error(self):
        """Test SQL execution error in flatten_risk_records"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        
        transaction_executor.side_effect = Exception("SQL execution failed")
        
        with patch('your_module.logger') as mock_logger:
            await flatten_risk_records(transaction_executor, query_executor)
            
            mock_logger.info.assert_called_with("Error flattening records: SQL execution failed")

class TestRemoveHtmlTags:
    
    def test_remove_html_tags_success(self):
        """Test successful HTML tag removal"""
        html_text = "<p>This is <b>bold</b> and <i>italic</i> text</p>"
        
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = "This is bold and italic text"
            
            result = remove_html_tags(html_text)
            
            assert result == "This is bold and italic text"
            mock_soup.assert_called_once_with(html_text, "html.parser")

    def test_remove_html_tags_beautifulsoup_error(self):
        """Test BeautifulSoup parsing error with regex fallback"""
        html_text = "<p>Test content</p>"
        
        with patch('your_module.BeautifulSoup', side_effect=Exception("BeautifulSoup failed")), \
             patch('your_module.re.sub', return_value="Test content"), \
             patch('your_module.logger') as mock_logger:
            
            result = remove_html_tags(html_text)
            
            assert result == "Test content"
            mock_logger.info.assert_called_with("Error parsing HTML with BeautifulSoup: BeautifulSoup failed")

    def test_remove_html_tags_empty_input(self):
        """Test with empty input"""
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = ""
            
            result = remove_html_tags("")
            
            assert result == ""

    def test_remove_html_tags_none_input(self):
        """Test with None input"""
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = ""
            
            result = remove_html_tags(None)
            
            assert result == ""

    def test_remove_html_tags_complex_html(self):
        """Test with complex HTML structure"""
        complex_html = """
        <div class="content">
            <h1>Title</h1>
            <p>Paragraph with <a href="link">link</a></p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
        """
        
        with patch('your_module.BeautifulSoup') as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = "Title Paragraph with link Item 1 Item 2"
            
            result = remove_html_tags(complex_html)
            
            assert result == "Title Paragraph with link Item 1 Item 2"

# Integration-style tests
class TestIntegrationScenarios:
    
    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Test simulating the full workflow"""
        with patch('your_module.sync_activity_taxonomy', return_value=["Activity done"]), \
             patch('your_module.sync_control_taxonomy', return_value=["Control done"]), \
             patch('your_module.sync_risk_taxonomy', return_value=["Risk done"]), \
             patch('your_module.sync_root_cause_taxonomy', return_value=["Root cause done"]), \
             patch('your_module.sync_reg_inventory', return_value=["Regulation done"]), \
             patch('your_module.get_async_transaction_executor'), \
             patch('your_module.get_async_query_executor'), \
             patch('your_module.get_environment'), \
             patch('your_module.GrcServiceSettings'):
            
            result = await sync_attributes_to_database()
            
            assert "Activity done|Control done|Risk done|Root cause done|Regulation done" == result

# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, mock_environment, mock_credentials):
        """Test handling large datasets"""
        transaction_executor = AsyncMock()
        query_executor = AsyncMock()
        grc_service_config = MagicMock()
        
        # Create large dataset
        large_requirement_data = [
            {
                "reqId": f"REQ{i:03d}",
                "reqName": f"Requirement {i}",
                "requirementSummaryEnglishText": f"<p>Summary for requirement {i}</p>",
                "citation": f"Citation {i}",
                "functionList": [{"name": f"func{i}"}],
                "etr": f"etr{i}",
                "riskLibAnchoring": {"level": "medium"},
                "regulatory": f"regulatory{i}",
                "regulatoryTier": "Tier 2",
                "regulatoryCountry": "US",
                "inventoryType": f"type{i}",
                "keyContactInfo": {"contact": f"contact{i}"}
            }
            for i in range(1000)  # 1000 records
        ]
        
        query_executor.return_value = [{"country": "US"}]
        transaction_executor.return_value = MagicMock()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "requirementSelectionData": large_requirement_data
            }
        }
        
        with patch('your_module.get_environment', return_value=mock_environment), \
             patch('your_module.yaml.safe_load', return_value=mock_credentials), \
             patch('your_module.requests.post', return_value=mock_response), \
             patch('your_module.truncate_db', return_value="Truncated"), \
             patch('your_module.remove_html_tags', return_value="Summary"), \
             patch('your_module.json.dumps', return_value='{"test": "data"}'):
            
            result = await sync_reg_inventory(transaction_executor, query_executor, grc_service_config, mock_environment)
            
            assert "1000 records in eim.regulation_t1_new table" in str(result)

# Configuration for pytest
@pytest.fixture(autouse=True)
def mock_logger():
    """Auto-use logger mock for all tests"""
    with patch('your_module.logger') as mock_log:
        yield mock_log
