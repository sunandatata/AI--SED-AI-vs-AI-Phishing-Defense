
import whois
from datetime import datetime

def get_domain_age_days(domain):
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        
        # creation_date can be a list or a single datetime object
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
            
        if creation_date:
            age = datetime.now() - creation_date
            return age.days
        return None
    except Exception as e:
        print(f"Error checking {domain}: {e}")
        return None

test_domains = ["google.com", "microsoft.com", "comcast.com", "thisis-a-test-domain-123.net"]
for d in test_domains:
    age = get_domain_age_days(d)
    print(f"Domain: {d}, Age in days: {age}")
