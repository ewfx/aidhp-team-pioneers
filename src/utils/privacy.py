from typing import Dict, List, Any
import hashlib
import json
from datetime import datetime
import logging

class PrivacyManager:
    def __init__(self):
        self.consent_records = {}
        self.data_retention_policy = {
            'customer_data': 365,  # days
            'transaction_history': 730,  # days
            'interaction_logs': 90  # days
        }
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for privacy."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def record_consent(self, 
                      customer_id: str,
                      consent_type: str,
                      consent_status: bool,
                      timestamp: datetime = None) -> None:
        """Record customer consent for data processing."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if customer_id not in self.consent_records:
            self.consent_records[customer_id] = {}
        
        self.consent_records[customer_id][consent_type] = {
            'status': consent_status,
            'timestamp': timestamp.isoformat(),
            'version': '1.0'
        }
        
        logging.info(f"Consent recorded for customer {customer_id}: {consent_type} = {consent_status}")
    
    def check_consent(self, customer_id: str, consent_type: str) -> bool:
        """Check if customer has given consent for specific data processing."""
        if customer_id not in self.consent_records:
            return False
        
        if consent_type not in self.consent_records[customer_id]:
            return False
        
        return self.consent_records[customer_id][consent_type]['status']
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive customer data."""
        anonymized = data.copy()
        
        # Hash sensitive fields
        sensitive_fields = ['email', 'phone', 'address', 'ssn']
        for field in sensitive_fields:
            if field in anonymized:
                anonymized[field] = self.hash_sensitive_data(str(anonymized[field]))
        
        return anonymized
    
    def check_data_retention(self, data_type: str, timestamp: datetime) -> bool:
        """Check if data should be retained based on retention policy."""
        if data_type not in self.data_retention_policy:
            return True  # Default to retain if type not specified
        
        retention_days = self.data_retention_policy[data_type]
        age = (datetime.now() - timestamp).days
        
        return age <= retention_days
    
    def generate_privacy_report(self, customer_id: str) -> Dict[str, Any]:
        """Generate privacy report for a customer."""
        if customer_id not in self.consent_records:
            return {
                'status': 'error',
                'message': 'Customer not found'
            }
        
        report = {
            'customer_id': customer_id,
            'consent_status': self.consent_records[customer_id],
            'data_retention': {
                data_type: f"{days} days"
                for data_type, days in self.data_retention_policy.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return report
    
    def validate_compliance(self, 
                          data: Dict[str, Any],
                          regulations: List[str] = None) -> Dict[str, bool]:
        """Validate data processing against regulatory requirements."""
        if regulations is None:
            regulations = ['GDPR', 'CCPA', 'PCI-DSS']
        
        compliance_results = {}
        
        for regulation in regulations:
            if regulation == 'GDPR':
                compliance_results['GDPR'] = self._validate_gdpr(data)
            elif regulation == 'CCPA':
                compliance_results['CCPA'] = self._validate_ccpa(data)
            elif regulation == 'PCI-DSS':
                compliance_results['PCI-DSS'] = self._validate_pci_dss(data)
        
        return compliance_results
    
    def _validate_gdpr(self, data: Dict[str, Any]) -> bool:
        """Validate GDPR compliance."""
        required_fields = ['consent_status', 'data_processing_purpose']
        return all(field in data for field in required_fields)
    
    def _validate_ccpa(self, data: Dict[str, Any]) -> bool:
        """Validate CCPA compliance."""
        required_fields = ['opt_out_status', 'data_sale_consent']
        return all(field in data for field in required_fields)
    
    def _validate_pci_dss(self, data: Dict[str, Any]) -> bool:
        """Validate PCI-DSS compliance."""
        if 'payment_data' in data:
            return self.hash_sensitive_data(str(data['payment_data'])) != str(data['payment_data'])
        return True
    
    def export_consent_records(self, filepath: str) -> None:
        """Export consent records to file."""
        with open(filepath, 'w') as f:
            json.dump(self.consent_records, f, indent=2)
    
    def import_consent_records(self, filepath: str) -> None:
        """Import consent records from file."""
        with open(filepath, 'r') as f:
            self.consent_records = json.load(f) 