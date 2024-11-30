# SberProcessMining (SberPM) – Process Mining Python framework
SberPM is an open-source Python library for conducting a comprehensive analysis of business processes with the use of process mining and machine learning techniques. By implementing this tool, objective and deep insights into the process on all levels can be revealed. These insights are then used to detect problems such as bottlenecks and deviations and identify potential opportunities for process improvement and optimization.

With SberPM you can:
- **Discover the real process flow and create visualization of the generated model**
    Having the event log of a business process, you can build the process model in the form of a graph by using the algorithms called miners and visualize it. In order to do this, the library provides a number of algorithms, that is Simple Miner, Causal Miner, Heuristic Miner, Alpha/Alpha+ Miners, and Inductive Miner. Thus different process weaknesses such as bottlenecks, loops, and deviations can be identified.
- **Calculate process performance indicators**
    As a part of performance analysis, SberPM offers five basic types of metrics, each designed for a prticular object to group by. In fact, it can be metrics by IDs, unique traces (=sequences of activities), activities, transitions (=two consequent activities), or users.
- **Visualize process performance indicators**
    Once the process performance indicators are assessed, they can be either mapped on a data-driven process graph or plotted in a separate interactive figure. By doing this, it is possible to understand of how things are working and where shifts are possible.

SberPM Python library is being developed by the Sber Process Mining Team.

# Installation
### Installation via pip
```bash 
pip install sberpm
```
Additionally, you might need to install graphviz executables and add the path to the executables to PATH variable: https://graphviz.org/download/

# Examples
To find out how to work with SberPM, see [tutorials](https://pypi.org/project/sberpm/#files) in files tar.gz.

# Contacts
If you have any questions or suggestions, feel free to contact us!
- Библиотека Sber Process Mining ("SberPM_lib@sberbank.ru")

