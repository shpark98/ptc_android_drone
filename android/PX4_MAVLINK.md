# PX4 MAVLink module

`Px4MavlinkManager` provides the first phone-to-PX4 link layer:

- `connectUsb(device)` opens a granted Android USB Host CDC/bulk interface.
- `connectUdp(host, port)` opens a MAVLink UDP link (default port `14540`).
- `Listener.onMessage` receives decoded MAVLink messages.
- `sendMessage(...)` sends an explicitly selected MAVLink message.
- A 1 Hz ground-station heartbeat is sent while connected.

The manager does not arm the vehicle, switch flight modes, or send position
setpoints automatically. Those operations should be added behind an explicit
operator action and a link-loss failsafe. USB permission must be requested with
`UsbManager.requestPermission()` before calling `connectUsb`.

Example:

```kotlin
val px4 = Px4MavlinkManager(this, lifecycleScope, object : Px4MavlinkManager.Listener {
    override fun onLinkState(state: Px4MavlinkManager.State, detail: String) { /* update UI */ }
    override fun onMessage(message: MavlinkMessage<*>) { /* telemetry */ }
    override fun onError(error: Throwable) { /* log */ }
})
px4.connectUdp("192.168.1.10")
// or, after USB permission: px4.connectUsb(usbDevice)
```
