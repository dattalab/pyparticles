behaviors = [(0, "f"),
(9, "r"),
(42, "f"),
(56, "i"),
(70, "r"),
(78, "f"),
(94, "r"),
(122, "r"),
(144, "f"),
(161, "r"),
(177, "f"),
(194, "i"),
(204, "l"),
(231, "f"),
(349, "j"),
(381, "f"),
(388, "l"),
(407, "f"),
(420, "r"),
(435, "f"),
(472, "i"),
(478, "r"),
(499, "l"),
(512, "f"),
(521, "i"),
(557, "f"),
(571, "r"),
(590, "f"),
(609, "l"),
(615, "f"),
(638, "l"),
(650, "f"),
(688, "l"),
(697, "f"),
(701, "l"),
(709, "f"),
(719, "r"),
(741, "i"),
(771, "f"),
(790, "s"),
(826, "f"),
(851, "i"),
(861, "r"),
(889, "f"),
(974, "r"),
(994, "")]

string_labels = np.zeros((behaviors[-1][0],), 'string')
for i in range(len(behaviors)-1):
    idx = np.r_[behaviors[i][0]:behaviors[i+1][0]]
    string_labels[idx] = behaviors[i][1]

unique_behaviors = np.unique(string_labels)
labels = np.zeros((len(string_labels),), 'int')
for i,l in enumerate(unique_behaviors):
	idx = string_labels == l
	labels[idx] = i


human_hash = {
	"f":"forward",
	"i":"investigate",
	"j":"junk",
	"l":"turn left",
	"r":"turn right",
	"s":"sniff"
}

machine_hash = {}
for l in unique_behaviors:
	machine_hash[l] = np.argwhere(unique_behaviors == l)[0][0]

