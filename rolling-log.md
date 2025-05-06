# CTL: Rolling log

## before April 2025 CTL:
what has happend where are we going

## April 2025: week 1
- calico *.*.3 doesnt work and has a issue with creating the calico-typha containers which breaks the whole system since it is not able to interact with storage and get configutation files.

- Writing ansible playbook to reproduce the cluster configuration.

- Problem was a interface made by nerdctl for connecting and running containers. without my knowledge. The Bird BGP auto ip detect procedure inside the calico-nodes detected this as a pod network where they should have access and atempted to connect to it. however this caused it to continually wait for a responce and not establish any other connections with working networks. 

- working to create a persistant volume claim for the monitoring service..

## April 2025: week 2
- sdn paper and review
- debugging rpc and knative issues

## April 2025: week 3
- Getting knative working and starting monitoring/debugging/performance testing

## April 2025: week 4
- monitoring research
- initial tuning and volleball
- USE method, TSA method
- powermeasurements of the display configuration
- locust