
PATH = "."

KUBE_URL = "https://localhost:6443" 
KUBE_API_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjBPVzlCOUpEQnpodFZpeV91YVZ3MkJHdlhfU3ItX3VUV0h0cjF6d0prSUEifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF1bHQtdG9rZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmYXVsdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImVjM2NhNjEzLTFjMGQtNDQ3Zi1hMDJlLTcyMTcyMDQ0MWJiZCIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZWZhdWx0OmRlZmF1bHQifQ.ZmRvd5iJWlcQ9o9aSL2mLWVUW9Zz61KEYyouGXPaZn9zMHiVqApXmGmTupGn8agoLC5zoFzFGy2x_25yHFMWieE9R7sBdyMwg8YoqycHe89NtjenP3Hl_NNSNPe-uuILFVhq2dRC2X6ByrlTIlf-MMGVB1f9sj3VPjg9Mtyq95jeM4M52rmAQqVBoXkvL_AwyzkzcLv54bLvi1Ysc32IO2aOCWneDw1ueAeFlviEKwCpmUTjyuqL9je250JeLqyiEFpWgnjHiwxYlhavm9mClbNXnBr3FNaJZ35OkGFT0zBy1seF-V_ihhnLX_60evPnFYVT5CwtCOTMYoglgQv2tQ"


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
