from __future__ import division

dv = None
def go_parallel(scenefile_path_relative_to_engines,behavior_data_path_relative_to_engines,frame_range):
    global dv
    if dv is None:
        from IPython.parallel import Client
        c = Client()
        dv = c[:]

        codestr = \
'''
try:
    print ms
except:
    import experiments
    ms = experiments._build_mousescene("%(scenefile)s")
    images, xytheta = experiments._load_data("%(datapath)s",%(frame_range)s)
''' % {'scenefile':scenefile_path_relative_to_engines,'datapath':behavior_data_path_relative_to_engines,'frame_range':frame_range}

        print codestr
        dv.execute(codestr,block=True)

    return dv

