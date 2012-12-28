clusters = cmds.ls(type='skinCluster')
skinCluster_name = clusters[0]

# Get the influenced geometry (just a name)
geometry = cmds.skinCluster(skinCluster_name, query=True, geometry=True)
geometry_name = geometry[0]

# Get the transforms that influence the geometry (the names of the joints)
influences = cmds.skinCluster(skinCluster_name, query=True, weightedInfluence=True)

rotations = np.zeros((len(influences), 3))
for i, influence in enumerate(influences):
    rotations[i] = cmds.xform(influence, rotation=True, query=True)
