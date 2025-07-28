
PATH = "./experiments/5-closing-time/"

KUBE_URL = "https://localhost:6443" 
KUBE_API_TOKEN =  ("eyJhbGciOiJSUzI1NiIsImtpZCI6ImtQQmZaNXU4all2UEVTMWg3Q0lwdmpNcEFLYVdZY2c4S0Y0d3NaRXZyMUkifQ"
                   ".eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uY"
                   "W1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF"
                   "1bHQtdG9rZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmY"
                   "XVsdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImEwOTA0OGRiLTM"
                   "1OTItNGQ2OS1hMmY3LWU5MDM0Y2M1Y2U2NyIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZWZhdWx0OmRlZ"
                   "mF1bHQifQ.yXjIcIA4uL4gASdj5TE1v4hiHAKROka58RG7MAJOlXGQkekYVTkzc3rM0tmOMClEEED9kAZTb6Tdp1u2"
                   "2b9zBs4Qoil58wQ-ehlH4HMJG573lPRBVoq5kB3l3rKpwsWsfXpqm4xvWxlBxT6tZT9UZNTTcYUkLjnshGwsaE55NZ"
                   "M7TG4YIeslRj3CAT-gOGWFQxIt1QFhZUor3D6JHrDznLPYB5iAeqXbOuvPClILtlmXoVftp3hmOGtlYwH8uWox9YHI"
                   "_WECYrYEXp92a7yn7iq953hwD2lp22vozPDb_a5cTxC3AaSqv9FUTRby-fS8bpuXhKuyd-yLe9YqZTzk8Q")

GEN_API_URL = "http://localhost:4242/generate"

SPAN_PROCESS_MAP = {'HTTP GET /hotels': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',
'HTTP GET /recommendations': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',
'HTTP POST /user': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',
'HTTP POST /reservation': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',

'memcached_get_profile': './experiments/4-gen-doc/yaml/memcached-profile-deployment.yaml',
'memcached_capacity_get_multi_number': './experiments/4-gen-doc/yaml/memcached-reservation-deployment.yaml',
'memcached_reserve_get_multi_number': './experiments/4-gen-doc/yaml/memcached-reservation-deployment.yaml',
'memcached_get_multi_rate': './experiments/4-gen-doc/yaml/memcached-rate-deployment.yaml',

'/profile.Profile/GetProfiles': './experiments/4-gen-doc/yaml/srv-profile.yaml',
'/search.Search/Nearby': './experiments/4-gen-doc/yaml/srv-search.yaml',
'/user.User/CheckUser': './experiments/4-gen-doc/yaml/srv-user.yaml',
'/geo.Geo/Nearby': './experiments/4-gen-doc/yaml/srv-geo.yaml',
'/recommendation.Recommendation/GetRecommendations': './experiments/4-gen-doc/yaml/srv-recommendation.yaml',
'/rate.Rate/GetRates': './experiments/4-gen-doc/yaml/srv-rate.yaml',
'/reservation.Reservation/CheckAvailability': './experiments/4-gen-doc/yaml/srv-reservation.yaml',
'/reservation.Reservation/MakeReservation': './experiments/4-gen-doc/yaml/srv-reservation.yaml'}
