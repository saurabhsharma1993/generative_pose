import numpy as np
import src.utils as utils
import matplotlib.pyplot as plt

edge_colors = {}
#right lower body
edge_colors[(0,1)] = 'xkcd:purple'
edge_colors[(1,2)] = 'xkcd:purple'
edge_colors[(2,3)] = 'xkcd:purple'
#left lower body
edge_colors[(0,6)] = 'xkcd:orange'
edge_colors[(6,7)] = 'xkcd:orange'
edge_colors[(7,8)] = 'xkcd:orange'
#torso
edge_colors[(0,12)] = 'xkcd:black'
edge_colors[(12,13)] = 'xkcd:black'
edge_colors[(13,14)] = 'xkcd:black'
edge_colors[(14,15)] = 'xkcd:black'
#right upper body
edge_colors[(13,25)] = 'xkcd:blue'
edge_colors[(25,26)] = 'xkcd:blue'
edge_colors[(26,27)] = 'xkcd:blue'
#left upper body
edge_colors[(13,17)] = 'xkcd:crimson'
edge_colors[(17,18)] = 'xkcd:crimson'
edge_colors[(18,19)] = 'xkcd:crimson'

def show3Dpose(title, channels_pred, add_labels=False):

  fig = plt.figure()
  ax = fig.add_subplot((111), projection='3d')

  assert channels_pred.size == len(utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels_pred.size

  vals_pred = np.reshape( channels_pred, (len(utils.H36M_NAMES), -1) )

  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  z, zorder = np.zeros(16), np.arange(0,16)
  for i in np.arange(len(I)):
      z[i] = -(vals_pred[I[i], 2] + vals_pred[J[i], 2]) / 2
  zorder = np.argsort(z)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [-np.array([vals_pred[I[i], j], vals_pred[J[i], j]]) for j in range(3)]
    ax.plot(-x, -z, y, linewidth=2, c=edge_colors[(I[i], J[i])], zorder=zorder[i])

  RADIUS = 500 # space around the subject
  xroot, yroot, zroot = vals_pred[0,0], vals_pred[0,2], -vals_pred[0,1]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

  fig.set_dpi(500)
  fig.savefig(title, bbox_inches='tight')

def show2Dpose(title, channels_pred, lcolor="xkcd:darkblue", rcolor="xkcd:crimson", add_labels=False):

  fig = plt.figure()
  ax = fig.add_subplot((111))

  assert channels_pred.size == len(utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels_pred.size
  vals_pred = np.reshape( channels_pred, (len(utils.H36M_NAMES), -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [-np.array( [vals_pred[I[i], j], vals_pred[J[i], j]] ) for j in range(2)]
    ax.plot(-x, y, lw=2, c=lcolor if LR[i] else rcolor, linewidth=4)

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

  ax.set_aspect('equal')

  fig.set_dpi(500)
  fig.savefig(title + '.png', bbox_inches='tight')
