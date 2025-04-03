# CTL: Rolling log

## before April 2025 CTL:
what has happend where are we going

## April 2025: week 1
- calico *.*.3 doesnt work and has a issue with creating the calico-typha containers which breaks the whole system since it is not able to interact with storage and get configutation files.

- Writing ansible playbook to reproduce the cluster configuration.

- Problem was a interface made by nerdctl for connecting and running containers. without my knowledge. The Bird BGP auto ip detect procedure inside the calico-nodes detected this as a pod network where they should have access and atempted to connect to it. however this caused it to continually wait for a responce and not establish any other connections with working networks. 

## April 2025: week 2

## April 2025: week 3

## April 2025: week 4
