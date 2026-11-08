Dim objShell, objShortcut
Set objShell = CreateObject("WScript.Shell")
Set objShortcut = objShell.CreateShortcut("C:\Users\Lenovo\Desktop\Wink HD Enhancer.lnk")
objShortcut.TargetPath = "C:\Users\Lenovo\.gemini\antigravity\scratch\wink_hd_enhancer\run_app.bat"
objShortcut.WorkingDirectory = "C:\Users\Lenovo\.gemini\antigravity\scratch\wink_hd_enhancer"
objShortcut.IconLocation = "C:\Users\Lenovo\.gemini\antigravity\scratch\wink_hd_enhancer\app\static\assets\app_icon.ico,0"
objShortcut.Description = "Wink HD Enhancer - AI Super-Resolution Studio"
objShortcut.Save
WScript.Echo "Shortcut updated successfully!"
