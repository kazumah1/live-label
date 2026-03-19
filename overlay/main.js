const { app, BrowserWindow, ipcMain } = require('electron');

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 360,
    height: 290,
    minWidth: 200,
    minHeight: 140,
    frame: false,
    transparent: false,
    alwaysOnTop: true,
    resizable: true,
    hasShadow: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // Invisible to CGWindowListCreateImage — pyautogui won't see this window
  win.setContentProtection(true);

  // 'screen-saver' level floats above everything, including fullscreen apps
  win.setAlwaysOnTop(true, 'screen-saver');

  // Read port from CLI args: --port 5001
  const portArg = process.argv.find(a => a.startsWith('--port='));
  const port = portArg ? portArg.split('=')[1] : '5000';

  win.loadFile('index.html', { query: { port } });
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());

// Window dragging via IPC (frame: false means no native drag)
ipcMain.on('drag-start', () => {});

ipcMain.on('drag-delta', (_, { dx, dy }) => {
  if (!win) return;
  const [x, y] = win.getPosition();
  win.setPosition(x + dx, y + dy);
});

ipcMain.on('close-window', () => {
  if (win) win.close();
});
