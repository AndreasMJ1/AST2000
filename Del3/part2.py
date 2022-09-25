from part1 import *

### Velger planet 2 

sqr_m_solarpanel1 = (40/0.12)/flux_rec(np.linalg.norm([utils.AU_to_m(p_pos[0,2]),utils.AU_to_m(p_pos[1,2])]))

print(sqr_m_solarpanel1)