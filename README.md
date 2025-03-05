# closing-the-loop
a repository for researching and experimenting with applying large language model (LLM) based agents to cloud-edge orchestration tooling to resolve performance anomalies.

TODO: add figure to illustrate the general concept.
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

TODO: add figure to give an overview of the causes and mitigation tactics for performance anomalies.
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

# background literature
This is an overview of the project's background literature.

## Performance anomalies
1. [Guerron X, Abrahao S, Insfran E, Fernandez-Diego M, Gonzalez-Ladron-De-Guevara F. A Taxonomy of Quality Metrics for Cloud Services. IEEE Access. 2020;8:131461–98.](https://ieeexplore.ieee.org/document/9139920)
2. [Zoure M, Ahmed T, Reveillere L. Network Services Anomalies in NFV: Survey, Taxonomy, and Verification Methods. IEEE Trans Netw Serv Manage. 2022 Jun;19(2):1567–84.](https://ieeexplore.ieee.org/document/9686060)
3. [Ibidunmoye O, Hernández-Rodriguez F, Elmroth E. Performance Anomaly Detection and Bottleneck Identification. ACM Comput Surv. 2015 Sep 29;48(1):1–35.](https://dl.acm.org/doi/10.1145/2791120)
4. [Moghaddam SK, Buyya R, Ramamohanarao K. Performance-Aware Management of Cloud Resources: A Taxonomy and Future Directions. ACM Comput Surv. 2020 Jul 31;52(4):1–37.](https://dl.acm.org/doi/10.1145/3337956)

## Benchmarks
1. Y. Gan et al., “An Open-Source Benchmark Suite for Microservices and Their Hardware-Software Implications for Cloud & Edge Systems,” in Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, Providence RI USA: ACM, Apr. 2019, pp. 3–18. doi: 10.1145/3297858.3304013.
2. M. Grambow, T. Pfandzelter, L. Burchard, C. Schubert, M. Zhao, and D. Bermbach, “BeFaaS: An Application-Centric Benchmarking Framework for FaaS Platforms,” Nov. 01, 2021, arXiv: arXiv:2102.12770. Accessed: Nov. 08, 2024. [Online]. Available: http://arxiv.org/abs/2102.12770
3. K. R. Rajput, C. D. Kulkarni, B. Cho, W. Wang, and I. K. Kim, “EdgeFaaSBench: Benchmarking Edge Devices Using Serverless Computing,” in 2022 IEEE International Conference on Edge Computing and Communications (EDGE), Barcelona, Spain: IEEE, Jul. 2022, pp. 93–103. doi: 10.1109/EDGE55608.2022.00024.
4. P. Maissen, P. Felber, P. Kropf, and V. Schiavoni, “FaaSdom: a benchmark suite for serverless computing,” in Proceedings of the 14th ACM International Conference on Distributed and Event-based Systems, Montreal Quebec Canada: ACM, Jul. 2020, pp. 73–84. doi: 10.1145/3401025.3401738.
5. M. Copik, A. Calotoiu, P. Zhou, K. Taranov, and T. Hoefler, “FaaSKeeper: Learning from Building Serverless Services with ZooKeeper as an Example,” in Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing, Jun. 2024, pp. 94–108. doi: 10.1145/3625549.3658661.
6. M. Copik, A. Calotoiu, R. Bruno, G. Rethy, R. Böhringer, and T. Hoefler, “Process-as-a-Service: Elastic and Stateful Serverless with Cloud Processes.”.
7. M. Copik, G. Kwasniewski, M. Besta, M. Podstawski, and T. Hoefler, “SeBS: A Serverless Benchmark Suite for Function-as-a-Service Computing,” Jul. 02, 2021, arXiv: arXiv:2012.14132. Accessed: Nov. 08, 2024. [Online]. Available: http://arxiv.org/abs/2012.14132
8. T. Hoefler et al., “XaaS: Acceleration as a Service to Enable Productive High-Performance Cloud Computing,” Jan. 09, 2024, arXiv: arXiv:2401.04552. Accessed: Nov. 08, 2024. [Online]. Available: http://arxiv.org/abs/2401.04552

### Beyond
1. V. Anand, D. Garg, A. Kaufmann, and J. Mace, “Blueprint: A Toolchain for Highly-Reconfigurable Microservice Applications,” in Proceedings of the 29th Symposium on Operating Systems Principles, Koblenz Germany: ACM, Oct. 2023, pp. 482–497. doi: 10.1145/3600006.3613138.


## Anomaly Detection
1. T. Bohne, A.-K. P. Windler, and M. Atzmueller, “A Neuro-Symbolic Approach for Anomaly Detection and Complex Fault Diagnosis Exemplified in the Automotive Domain,” in Proceedings of the 12th Knowledge Capture Conference 2023, Pensacola FL USA: ACM, Dec. 2023, pp. 35–43. doi: 10.1145/3587259.3627546
2. J. H. de Souza Pereira, S. T. Kofuji, and P. F. Rosa, “Distributed Systems Ontology,” in 2009 3rd International Conference on New Technologies, Mobility and Security, Dec. 2009, pp. 1–5. doi: 10.1109/NTMS.2009.5384822.
3. H. Birkholz and I. Sieverdingbeck, “Improving root cause failure analysis in virtual networks via the interconnected-asset ontology,” in Proceedings of the Conference on Principles, Systems and Applications of IP Telecommunications, Chicago Illinois: ACM, Oct. 2014, pp. 1–8. doi: 10.1145/2670386.2670395.
4. A. Petrovska, S. Quijano, I. Gerostathopoulos, and A. Pretschner, “Knowledge aggregation with subjective logic in multi-agent self-adaptive cyber-physical systems,” in Proceedings of the IEEE/ACM 15th International Symposium on Software Engineering for Adaptive and Self-Managing Systems, Seoul Republic of Korea: ACM, Jun. 2020, pp. 149–155. doi: 10.1145/3387939.3391600.
5. Y. Zhang, Z. Chen, L. Guo, Y. Xu, W. Zhang, and H. Chen, “Making Large Language Models Perform Better in Knowledge Graph Completion,” in Proceedings of the 32nd ACM International Conference on Multimedia, Melbourne VIC Australia: ACM, Oct. 2024, pp. 233–242. doi: 10.1145/3664647.3681327.

## Root Cause Analysis
1. T. Wang and G. Qi, “A Comprehensive Survey on Root Cause Analysis in (Micro) Services: Methodologies, Challenges, and Trends,” Jul. 23, 2024, arXiv: arXiv:2408.00803. doi: 10.48550/arXiv.2408.00803.
2. H. X. Nguyen, S. Zhu, and M. Liu, “A Survey on Graph Neural Networks for Microservice-Based Cloud Applications,” Sensors, vol. 22, no. 23, p. 9492, Dec. 2022, doi: 10.3390/s22239492.
3. J. Soldani and A. Brogi, “Anomaly Detection and Failure Root Cause Analysis in (Micro) Service-Based Cloud Applications: A Survey,” ACM Comput. Surv., vol. 55, no. 3, pp. 1–39, Mar. 2023, doi: 10.1145/3501297.

### Primary Studies
1. M. S. Islam, W. Pourmajidi, L. Zhang, J. Steinbacher, T. Erwin, and A. Miranskyy, “Anomaly Detection in a Large-Scale Cloud Platform,” in 2021 IEEE/ACM 43rd International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), Madrid, ES: IEEE, May 2021, pp. 150–159. doi: 10.1109/ICSE-SEIP52600.2021.00024.
2. J. Hochenbaum, O. S. Vallis, and A. Kejariwal, “Automatic Anomaly Detection in the Cloud Via Statistical Learning,” Apr. 24, 2017, arXiv: arXiv:1704.07706. Accessed: Oct. 10, 2024. [Online]. Available: http://arxiv.org/abs/1704.07706
3. G. Somashekar, A. Dutt, R. Vaddavalli, S. B. Varanasi, and A. Gandhi, “B-MEG: Bottlenecked-Microservices Extraction Using Graph Neural Networks,” in Companion of the 2022 ACM/SPEC International Conference on Performance Engineering, Bejing China: ACM, Jul. 2022, pp. 7–11. doi: 10.1145/3491204.3527494.
4. M. Shetty et al., “Building AI Agents for Autonomous Clouds: Challenges and Design Principles,” Jul. 31, 2024, arXiv: arXiv:2407.12165. Accessed: Oct. 17, 2024. [Online]. Available: http://arxiv.org/abs/2407.12165
5. J. Kaldor et al., “Canopy: An End-to-End Performance Tracing And Analysis System,” in Proceedings of the 26th Symposium on Operating Systems Principles, Shanghai China: ACM, Oct. 2017, pp. 34–50. doi: 10.1145/3132747.3132749.
6. M. Li et al., “Causal Inference-Based Root Cause Analysis for Online Service Systems with Intervention Recognition,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington DC USA: ACM, Aug. 2022, pp. 3230–3240. doi: 10.1145/3534678.3539041.
7. P. Wang et al., “CloudRanger: Root Cause Identification for Cloud Native Systems,” in 2018 18th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGRID), Washington, DC, USA: IEEE, May 2018, pp. 492–502. doi: 10.1109/CCGRID.2018.00076.
8. Y. Zhang et al., “CloudRCA: A Root Cause Analysis Framework for Cloud Computing Platforms,” in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, Virtual Event Queensland Australia: ACM, Oct. 2021, pp. 4373–4382. doi: 10.1145/3459637.3481903.
9. C. Zhang et al., “DeepTraLog: Trace-Log Combined Microservice Anomaly Detection through Graph-based Deep Learning,” in 2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE), May 2022, pp. 623–634. doi: 10.1145/3510003.3510180.
10. Y. Zhao, Y. Zheng, H. Luo, D. Wei, C. Liu, and K. Chen, “Design and Implement of AIOps System Based on Knowledge Graph,” in 2023 5th International Conference on Electronics and Communication, Network and Computer Technology (ECNCT), Guangzhou, China: IEEE, Aug. 2023, pp. 285–288. doi: 10.1109/ECNCT59757.2023.10281148.
11. A. Gulenko, F. Schmidt, A. Acker, M. Wallschlager, O. Kao, and F. Liu, “Detecting Anomalous Behavior of Black-Box Services Modeled with Distance-Based Online Clustering,” in 2018 IEEE 11th International Conference on Cloud Computing (CLOUD), San Francisco, CA, USA: IEEE, Jul. 2018, pp. 912–915. doi: 10.1109/CLOUD.2018.00134
12. H. Jayathilaka, C. Krintz, and R. Wolski, “Detecting Performance Anomalies in Cloud Platform Applications,” IEEE Trans. Cloud Comput., vol. 8, no. 3, pp. 764–777, Jul. 2020, doi: 10.1109/TCC.2018.2808289.
13. W. Lv et al., “Graph-Reinforcement-Learning-Based Dependency-Aware Microservice Deployment in Edge Computing,” IEEE Internet of Things Journal, vol. 11, no. 1, pp. 1604–1615, Jan. 2024, doi: 10.1109/JIOT.2023.3289228.
14. J. Chen, H. Huang, and H. Chen, “Informer: irregular traffic detection for containerized microservices RPC in the real world,” in Proceedings of the 4th ACM/IEEE Symposium on Edge Computing, in SEC ’19. New York, NY, USA: Association for Computing Machinery, Nov. 2019, pp. 389–394. doi: 10.1145/3318216.3363375.
15. T. Wittkopp, P. Wiesner, and O. Kao, “LogRCA: Log-based Root Cause Analysis for Distributed Services,” May 22, 2024, arXiv: arXiv:2405.13599. Accessed: Oct. 10, 2024. [Online]. Available: http://arxiv.org/abs/2405.13599
16. L. Wu, J. Tordsson, E. Elmroth, and O. Kao, “MicroRCA: Root Cause Localization of Performance Issues in Microservices,” in NOMS 2020 - 2020 IEEE/IFIP Network Operations and Management Symposium, Budapest, Hungary: IEEE, Apr. 2020, pp. 1–9. doi: 10.1109/NOMS47738.2020.9110353.
17. L. Wu, J. Tordsson, E. Elmroth, and O. Kao, “MicroRCA: Root Cause Localization of Performance Issues in Microservices,” in NOMS 2020 - 2020 IEEE/IFIP Network Operations and Management Symposium, Budapest, Hungary: IEEE, Apr. 2020, pp. 1–9. doi: 10.1109/NOMS47738.2020.9110353.
18. J. Lin, P. Chen, and Z. Zheng, “Microscope: Pinpoint Performance Issues with Causal Graphs in Micro-service Environments,” in Service-Oriented Computing, C. Pahl, M. Vukovic, J. Yin, and Q. Yu, Eds., Cham: Springer International Publishing, 2018, pp. 3–20. doi: 10.1007/978-3-030-03596-9_1.
19. C. Lin and H. Khazaei, “Modeling and Optimization of Performance and Cost of Serverless Applications,” IEEE Trans. Parallel Distrib. Syst., vol. 32, no. 3, pp. 615–632, Mar. 2021, doi: 10.1109/TPDS.2020.3028841.
20. J. Schoenfisch, C. Meilicke, J. V. Stülpnagel, J. Ortmann, and H. Stuckenschmidt, “Root cause analysis in IT infrastructures using ontologies and abduction in Markov Logic Networks,” Information Systems, vol. 74, pp. 103–116, May 2018, doi: 10.1016/j.is.2017.11.003.
21. J. Thalheim et al., “Sieve: actionable insights from monitored metrics in distributed systems,” in Proceedings of the 18th ACM/IFIP/USENIX Middleware Conference, Las Vegas Nevada: ACM, Dec. 2017, pp. 14–27. doi: 10.1145/3135974.3135977.
22. J. Thalheim et al., “Sieve: Actionable Insights from Monitored Metrics in Microservices,” Sep. 20, 2017, arXiv: arXiv:1709.06686. Accessed: Nov. 21, 2024. [Online]. Available: http://arxiv.org/abs/1709.06686
23. F. Giraldeau and M. Dagenais, “Wait Analysis of Distributed Systems Using Kernel Tracing,” IEEE Trans. Parallel Distrib. Syst., vol. 27, no. 8, pp. 2450–2461, Aug. 2016, doi: 10.1109/TPDS.2015.2488629.

### Causality Modeling
1. Z. Yao et al., “Chain-of-Event: Interpretable Root Cause Analysis for Microservices through Automatically Learning Weighted Event Causal Graph,” in Companion Proceedings of the 32nd ACM International Conference on the Foundations of Software Engineering, Porto de Galinhas Brazil: ACM, Jul. 2024, pp. 50–61. doi: 10.1145/3663529.3663827.

## Mitigation
1. V. U. Gsteiger, P. H. (Daniel) Long, Y. (Jerry) Sun, P. Javanrood, and M. Shahrad, “Caribou: Fine-Grained Geospatial Shifting of Serverless Applications for Sustainability,” in Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, Austin TX USA: ACM, Nov. 2024, pp. 403–420. doi: 10.1145/3694715.3695954
2. H. Qiu, S. S. Banerjee, S. Jha, Z. T. Kalbarczyk, and R. K. Iyer, “FIRM: An Intelligent Fine-grained Resource Management Framework for SLO-Oriented Microservices”.
3. G. Somashekar, “Proposal: Performance Management of Large-Scale Microservices Applications”.

## Knowledge graphs
1. J. He, M. D. Ma, J. Fan, D. Roth, W. Wang, and A. Ribeiro, “GIVE: Structured Reasoning with Knowledge Graph Inspired Veracity Extrapolation,” Oct. 11, 2024, arXiv: arXiv:2410.08475. Accessed: Oct. 28, 2024. [Online]. Available: http://arxiv.org/abs/2410.08475
2. X. Su et al., “Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine,” Oct. 07, 2024, arXiv: arXiv:2410.04660. Accessed: Oct. 28, 2024. [Online]. Available: http://arxiv.org/abs/2410.04660
3. J. Sun et al., “Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph,” Mar. 24, 2024, arXiv: arXiv:2307.07697. Accessed: Oct. 28, 2024. [Online]. Available: http://arxiv.org/abs/2307.07697

## LLM's
1. D. Li et al., “S*: Test Time Scaling for Code Generation,” Feb. 20, 2025, arXiv: arXiv:2502.14382. doi: 10.48550/arXiv.2502.14382.
2. Y. Chen et al., “Automatic Root Cause Analysis via Large Language Models for Cloud Incidents,” in Proceedings of the Nineteenth European Conference on Computer Systems, Athens Greece: ACM, Apr. 2024, pp. 674–688. doi: 10.1145/3627703.3629553.


