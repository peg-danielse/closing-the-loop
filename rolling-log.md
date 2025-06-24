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

## May 2025: week 1
- Tracing debugging -> revert back to 1.62.0
- programming to remove consul from the system.
- srv-Reservation is broken and will not run without crashing limiting the mem search to 301 hotels per instance... otherwise more caching is nessisary. automated implementation difficult. But a valid option to look into considering our usecase and the posibility of re-/writing code with llm's 
- made plots of inital state of hotel-reservations. should we consider more applications? 

## May 2025: week 2
- anomaly detection in traces using Isolation forest and static QoS value...
- anomaly detection in requests using Isolation forest and static QoS value...

## may 2025: week 3
- analysis of feature importance in isolation forests.
- rudimentary rootcause analysis using shapely values.
- analysis of workload patterns.
- meeting on phd:
    1. make sure you finish in time.
    2. close the loop asap.
    3. workload generation should folow scaling patterns and climbing and periodic.
    4. write some overview of the research. (every week one paragraph)
        p. oint
        e. xplaination
        e. xample
        l. link
    -
        a. nd
        b. ut
        t. herefor

## may 2025: week 4 
- SoTA slides for the 9th and 14th months evaluation. (see notes.)

## june 2025: week 1 
- Trip to Milaan for IPDPS conference.
- Present the research

## june 2025: week 2 
- sick

## june 2025: week 3
- sick + festival. Getting back into it.

## june 2025: week 4
- Presentation of IPDPS.
- Working on compiling the anomaly document for LLM adaptation.
- Testing the system.