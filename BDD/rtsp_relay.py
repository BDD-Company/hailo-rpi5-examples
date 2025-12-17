#! /usr/bin/env python

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

class RTPRelayFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super().__init__()
        self.set_launch(
            '( udpsrc port=5004 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96,clock-rate=90000" '
            '! rtph264depay ! h264parse '
            '! rtph264pay name=pay0 pt=96 config-interval=1 )'
        )
        self.set_shared(True)

server = GstRtspServer.RTSPServer()
mounts = server.get_mount_points()
mounts.add_factory("/cam", RTPRelayFactory())
server.attach(None)

print("RTSP server ready at rtsp://<ip>:8554/cam")
GLib.MainLoop().run()
