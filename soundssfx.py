import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Rectangle
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
import scipy.signal as signal
import tempfile
import os

class SoundLayer:
    def __init__(self, layer_id):
        self.id = layer_id
        self.waveform = 'sine'
        self.frequency = 440.0
        self.amplitude = 0.5
        self.harmonics = 3
        self.mod_freq = 0.0
        self.mod_depth = 0.0
        self.active = True

class SFXGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.duration = 1.0
        self.n_samples = int(self.sample_rate * self.duration)
        self.t = np.linspace(0, self.duration, self.n_samples, False)
        
    def generate_layer(self, layer):
        if not layer.active:
            return np.zeros(self.n_samples)
            
        # Base waveform generation
        if layer.waveform == 'sine':
            base = np.sin(2 * np.pi * layer.frequency * self.t)
        elif layer.waveform == 'square':
            base = signal.square(2 * np.pi * layer.frequency * self.t)
        elif layer.waveform == 'sawtooth':
            base = signal.sawtooth(2 * np.pi * layer.frequency * self.t)
        elif layer.waveform == 'triangle':
            base = signal.sawtooth(2 * np.pi * layer.frequency * self.t, 0.5)
        elif layer.waveform == 'noise':
            base = np.random.uniform(-1, 1, self.n_samples)
        else:
            base = np.sin(2 * np.pi * layer.frequency * self.t)
        
        # Add harmonics
        harmonic_signal = np.zeros_like(base)
        for i in range(1, layer.harmonics + 1):
            harmonic_signal += np.sin(2 * np.pi * layer.frequency * i * self.t) / i
        
        # Combine base and harmonics
        combined = base + 0.3 * harmonic_signal
        
        # Apply amplitude modulation if enabled
        if layer.mod_freq > 0 and layer.mod_depth > 0:
            modulator = 1 + layer.mod_depth * np.sin(2 * np.pi * layer.mod_freq * self.t)
            combined *= modulator
        
        # Normalize and apply amplitude
        if np.max(np.abs(combined)) > 0:
            combined = combined / np.max(np.abs(combined))
        return layer.amplitude * combined

    def generate_mixed_sound(self, layers):
        mixed = np.zeros(self.n_samples)
        active_layers = [layer for layer in layers if layer.active]
        
        if not active_layers:
            return mixed
            
        for layer in active_layers:
            mixed += self.generate_layer(layer)
        
        # Normalize mixed signal to prevent clipping
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.8
        
        return mixed

class WaveformVisualizer(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = np.zeros(100)
        self.bind(size=self.update_graphics, pos=self.update_graphics)
        
    def update_data(self, data):
        if len(data) > 0:
            # Downsample to 100 points for visualization
            step = max(1, len(data) // 100)
            self.data = data[::step][:100]
            self.update_graphics()
    
    def update_graphics(self, *args):
        self.canvas.clear()
        if len(self.data) == 0:
            return
            
        with self.canvas:
            Color(0.2, 0.6, 1, 1)  # Blue color
            # Draw waveform
            points = []
            width = self.width
            height = self.height
            center_y = self.y + height / 2
            
            for i, value in enumerate(self.data):
                x = self.x + i * width / len(self.data)
                y = center_y + value * height / 2.5
                points.extend([x, y])
            
            if len(points) >= 4:
                Line(points=points, width=1.5)

class EnhancedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_down = ''
        self.background_color = (0.3, 0.5, 0.8, 1)
        self.bind(on_press=self.on_button_press)
        
    def on_button_press(self, instance):
        # Visual feedback on press
        original_color = self.background_color
        self.background_color = (0.5, 0.7, 1, 1)
        Clock.schedule_once(lambda dt: setattr(self, 'background_color', original_color), 0.1)

class LayerControl(BoxLayout):
    def __init__(self, layer, on_change, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.on_change = on_change
        self.orientation = 'vertical'
        self.padding = dp(10)
        self.spacing = dp(8)
        self.size_hint_y = None
        self.height = dp(320)
        
        # Header with layer info and controls
        header = BoxLayout(orientation='horizontal', size_hint_y=0.12, spacing=dp(5))
        header.add_widget(Label(text=f'Layer {layer.id}', size_hint_x=0.2, bold=True))
        
        self.active_toggle = ToggleButton(
            text='ACTIVE' if layer.active else 'INACTIVE',
            size_hint_x=0.3,
            state='down' if layer.active else 'normal',
            background_color=(0.2, 0.7, 0.3, 1) if layer.active else (0.7, 0.2, 0.2, 1)
        )
        self.active_toggle.bind(on_press=self.toggle_active)
        header.add_widget(self.active_toggle)
        
        delete_btn = EnhancedButton(text='×', size_hint_x=0.1, background_color=(0.8, 0.2, 0.2, 1))
        delete_btn.bind(on_press=self.delete_layer)
        header.add_widget(delete_btn)
        
        self.add_widget(header)
        
        # Waveform selection
        waveform_layout = BoxLayout(orientation='horizontal', size_hint_y=0.12)
        waveform_layout.add_widget(Label(text='Waveform:', size_hint_x=0.3))
        self.waveform_btn = EnhancedButton(
            text=layer.waveform.upper(),
            size_hint_x=0.7,
            background_color=(0.3, 0.3, 0.3, 1)
        )
        self.waveform_btn.bind(on_press=self.cycle_waveform)
        self.waveforms = ['sine', 'square', 'sawtooth', 'triangle', 'noise']
        self.waveform_index = self.waveforms.index(layer.waveform)
        waveform_layout.add_widget(self.waveform_btn)
        self.add_widget(waveform_layout)
        
        # Frequency control
        self.add_control('Frequency (Hz)', 50, 2000, layer.frequency, self.update_freq, 'freq')
        
        # Amplitude control
        self.add_control('Amplitude', 0, 1, layer.amplitude, self.update_amp, 'amp')
        
        # Harmonics control
        self.add_control('Harmonics', 0, 10, layer.harmonics, self.update_harm, 'harm', is_int=True)
        
        # Modulation frequency
        self.add_control('Mod Freq (Hz)', 0, 50, layer.mod_freq, self.update_mod_freq, 'mod_freq', is_int=True)
        
        # Modulation depth
        self.add_control('Mod Depth', 0, 1, layer.mod_depth, self.update_mod_depth, 'mod_depth')
    
    def add_control(self, label, min_val, max_val, value, callback, attr_name, is_int=False):
        layout = BoxLayout(orientation='horizontal', size_hint_y=0.12)
        layout.add_widget(Label(text=label, size_hint_x=0.3))
        
        slider = Slider(min=min_val, max=max_val, value=value, step=1 if is_int else 0.01)
        setattr(self, f'{attr_name}_slider', slider)
        slider.bind(value=callback)
        layout.add_widget(slider)
        
        val_label = Label(
            text=str(int(value) if is_int else f"{value:.2f}"),
            size_hint_x=0.2,
            color=(0.8, 0.9, 1, 1)
        )
        setattr(self, f'{attr_name}_label', val_label)
        layout.add_widget(val_label)
        
        self.add_widget(layout)
    
    def toggle_active(self, instance):
        self.layer.active = instance.state == 'down'
        instance.text = 'ACTIVE' if self.layer.active else 'INACTIVE'
        instance.background_color = (0.2, 0.7, 0.3, 1) if self.layer.active else (0.7, 0.2, 0.2, 1)
        self.on_change()
    
    def cycle_waveform(self, instance):
        self.waveform_index = (self.waveform_index + 1) % len(self.waveforms)
        self.layer.waveform = self.waveforms[self.waveform_index]
        instance.text = self.layer.waveform.upper()
        self.on_change()
    
    def update_freq(self, instance, value):
        self.layer.frequency = value
        self.freq_label.text = str(int(value))
        self.on_change()
    
    def update_amp(self, instance, value):
        self.layer.amplitude = value
        self.amp_label.text = f"{value:.2f}"
        self.on_change()
    
    def update_harm(self, instance, value):
        self.layer.harmonics = int(value)
        self.harm_label.text = str(int(value))
        self.on_change()
    
    def update_mod_freq(self, instance, value):
        self.layer.mod_freq = value
        self.mod_freq_label.text = str(int(value))
        self.on_change()
    
    def update_mod_depth(self, instance, value):
        self.layer.mod_depth = value
        self.mod_depth_label.text = f"{value:.2f}"
        self.on_change()
    
    def delete_layer(self, instance):
        # This will be handled by the parent
        pass

class SFXApp(App):
    def build(self):
        self.generator = SFXGenerator()
        self.layers = []
        self.current_sound = None
        self.temp_file_path = None
        self.layer_counter = 0
        self.current_preset = None
        
        # Main layout with dark theme
        main_layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(15))
        main_layout.canvas.before.clear()
        with main_layout.canvas.before:
            Color(0.15, 0.15, 0.18, 1)  # Dark background
            self.bg_rect = Rectangle(pos=main_layout.pos, size=main_layout.size)
        main_layout.bind(size=self._update_rect, pos=self._update_rect)
        
        # Header
        header = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        header.add_widget(Label(
            text='PROFESSIONAL SFX GENERATOR',
            font_size=dp(24),
            bold=True,
            color=(0.8, 0.9, 1, 1)
        ))
        main_layout.add_widget(header)
        
        # Content area
        content = BoxLayout(orientation='horizontal', spacing=dp(15))
        
        # Left panel - Presets and Layers
        left_panel = BoxLayout(orientation='vertical', size_hint_x=0.4)
        
        # Presets section
        presets_panel = BoxLayout(orientation='vertical', size_hint_y=0.4)
        presets_header = BoxLayout(orientation='horizontal', size_hint_y=0.15)
        presets_header.add_widget(Label(
            text='SOUND PRESETS',
            font_size=dp(16),
            bold=True,
            color=(0.7, 0.8, 1, 1)
        ))
        presets_panel.add_widget(presets_header)
        
        # Preset buttons grid
        presets_grid = GridLayout(cols=2, spacing=dp(10), padding=dp(10))
        self.preset_buttons = []
        preset_data = [
            ('🔫 GUNFIRE', self.load_gunfire_preset),
            ('🌊 WATER', self.load_water_preset),
            ('🔧 METAL', self.load_metal_preset),
            ('🚂 TRAIN', self.load_train_preset)
        ]
        
        for text, callback in preset_data:
            btn = EnhancedButton(
                text=text,
                font_size=dp(14)
            )
            btn.bind(on_press=callback)
            self.preset_buttons.append(btn)
            presets_grid.add_widget(btn)
        
        presets_panel.add_widget(presets_grid)
        left_panel.add_widget(presets_panel)
        
        # Layers section
        layers_panel = BoxLayout(orientation='vertical')
        layers_header = BoxLayout(orientation='horizontal', size_hint_y=0.15, padding=(0, dp(5)))
        layers_header.add_widget(Label(
            text='SOUND LAYERS',
            font_size=dp(16),
            bold=True,
            color=(0.7, 0.8, 1, 1)
        ))
        add_layer_btn = EnhancedButton(
            text='+ ADD LAYER',
            size_hint_x=0.4,
            background_color=(0.3, 0.5, 0.8, 1)
        )
        add_layer_btn.bind(on_press=self.add_layer)
        layers_header.add_widget(add_layer_btn)
        layers_panel.add_widget(layers_header)
        
        # Layers scroll area
        self.layers_layout = GridLayout(cols=1, spacing=dp(10), size_hint_y=None)
        self.layers_layout.bind(minimum_height=self.layers_layout.setter('height'))
        
        scroll = ScrollView()
        scroll.add_widget(self.layers_layout)
        layers_panel.add_widget(scroll)
        left_panel.add_widget(layers_panel)
        
        content.add_widget(left_panel)
        
        # Right panel - Visualization and controls
        right_panel = BoxLayout(orientation='vertical', size_hint_x=0.6)
        
        # Visualization
        vis_header = BoxLayout(orientation='horizontal', size_hint_y=0.08)
        vis_header.add_widget(Label(
            text='WAVEFORM VISUALIZATION',
            font_size=dp(14),
            bold=True,
            color=(0.7, 0.8, 1, 1)
        ))
        right_panel.add_widget(vis_header)
        
        self.visualizer = WaveformVisualizer(size_hint_y=0.4)
        right_panel.add_widget(self.visualizer)
        
        # Controls
        controls = BoxLayout(orientation='vertical', size_hint_y=0.5, spacing=dp(15))
        
        # Play controls
        play_controls = BoxLayout(orientation='horizontal', size_hint_y=0.3, spacing=dp(15))
        
        self.play_btn = EnhancedButton(
            text='▶ PLAY',
            background_color=(0.2, 0.6, 0.3, 1),
            font_size=dp(16)
        )
        self.play_btn.bind(on_press=self.play_sound)
        play_controls.add_widget(self.play_btn)
        
        self.save_btn = EnhancedButton(
            text='💾 SAVE',
            background_color=(0.3, 0.5, 0.7, 1),
            font_size=dp(16)
        )
        self.save_btn.bind(on_press=self.save_sound)
        play_controls.add_widget(self.save_btn)
        
        controls.add_widget(play_controls)
        
        # Preset play controls
        preset_controls = BoxLayout(orientation='horizontal', size_hint_y=0.3, spacing=dp(15))
        preset_controls.add_widget(Label(text='Preset Sounds:', color=(0.7, 0.8, 1, 1)))
        
        self.preset_play_btn = EnhancedButton(
            text='▶ PLAY PRESET',
            background_color=(0.6, 0.4, 0.2, 1),
            font_size=dp(14)
        )
        self.preset_play_btn.bind(on_press=self.play_current_preset)
        preset_controls.add_widget(self.preset_play_btn)
        
        controls.add_widget(preset_controls)
        
        # Status
        self.status_label = Label(
            text='Select a preset or create layers',
            color=(0.7, 0.8, 1, 1),
            size_hint_y=0.1
        )
        controls.add_widget(self.status_label)
        
        right_panel.add_widget(controls)
        content.add_widget(right_panel)
        
        main_layout.add_widget(content)
        
        # Add initial layer
        self.add_layer(None)
        
        return main_layout
    
    def _update_rect(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size
    
    def add_layer(self, instance):
        self.layer_counter += 1
        new_layer = SoundLayer(self.layer_counter)
        self.layers.append(new_layer)
        
        layer_control = LayerControl(new_layer, self.on_layer_change)
        
        # Override delete button functionality
        layer_control.children[-1].children[0].bind(on_press=lambda x: self.delete_layer(layer_control))
        
        self.layers_layout.add_widget(layer_control)
        self.on_layer_change()
    
    def delete_layer(self, layer_control):
        if len(self.layers) <= 1:
            self.show_popup('Error', 'At least one layer required')
            return
            
        self.layers_layout.remove_widget(layer_control)
        self.layers.remove(layer_control.layer)
        self.on_layer_change()
    
    def on_layer_change(self):
        # Update visualization with mixed signal
        try:
            mixed_data = self.generator.generate_mixed_sound(self.layers)
            self.visualizer.update_data(mixed_data)
            active_count = len([l for l in self.layers if l.active])
            self.status_label.text = f'{active_count} active layers | Ready'
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
    
    def show_popup(self, title, message):
        popup = Popup(
            title=title,
            content=Label(text=message),
            size_hint=(0.6, 0.3)
        )
        popup.open()
    
    def highlight_preset_button(self, preset_name):
        # Reset all buttons
        for btn in self.preset_buttons:
            btn.background_color = (0.3, 0.5, 0.8, 1)
        
        # Highlight selected preset
        preset_map = {
            'gunfire': 0,
            'water': 1,
            'metal': 2,
            'train': 3
        }
        if preset_name in preset_map:
            self.preset_buttons[preset_map[preset_name]].background_color = (0.8, 0.6, 0.2, 1)
    
    # Preset loading functions
    def load_gunfire_preset(self, instance):
        self.current_preset = 'gunfire'
        self.highlight_preset_button('gunfire')
        
        # Clear existing layers
        self.clear_layers()
        
        # Layer 1: Sharp attack (noise)
        layer1 = SoundLayer(1)
        layer1.waveform = 'noise'
        layer1.frequency = 800
        layer1.amplitude = 0.9
        layer1.harmonics = 0
        layer1.mod_freq = 0
        layer1.mod_depth = 0
        layer1.active = True
        self.layers.append(layer1)
        self.layer_counter = 1
        
        # Layer 2: Decaying tone
        layer2 = SoundLayer(2)
        layer2.waveform = 'sawtooth'
        layer2.frequency = 200
        layer2.amplitude = 0.6
        layer2.harmonics = 5
        layer2.mod_freq = 20
        layer2.mod_depth = 0.7
        layer2.active = True
        self.layers.append(layer2)
        self.layer_counter = 2
        
        # Layer 3: Low frequency rumble
        layer3 = SoundLayer(3)
        layer3.waveform = 'sine'
        layer3.frequency = 60
        layer3.amplitude = 0.4
        layer3.harmonics = 2
        layer3.mod_freq = 0
        layer3.mod_depth = 0
        layer3.active = True
        self.layers.append(layer3)
        self.layer_counter = 3
        
        self.refresh_layers_ui()
        self.status_label.text = 'Loaded: Gunfire preset'
    
    def load_water_preset(self, instance):
        self.current_preset = 'water'
        self.highlight_preset_button('water')
        
        # Clear existing layers
        self.clear_layers()
        
        # Layer 1: Bubbling noise
        layer1 = SoundLayer(1)
        layer1.waveform = 'noise'
        layer1.frequency = 400
        layer1.amplitude = 0.5
        layer1.harmonics = 0
        layer1.mod_freq = 5
        layer1.mod_depth = 0.3
        layer1.active = True
        self.layers.append(layer1)
        self.layer_counter = 1
        
        # Layer 2: Flowing sound
        layer2 = SoundLayer(2)
        layer2.waveform = 'sine'
        layer2.frequency = 150
        layer2.amplitude = 0.3
        layer2.harmonics = 3
        layer2.mod_freq = 8
        layer2.mod_depth = 0.5
        layer2.active = True
        self.layers.append(layer2)
        self.layer_counter = 2
        
        # Layer 3: High frequency spray
        layer3 = SoundLayer(3)
        layer3.waveform = 'triangle'
        layer3.frequency = 1200
        layer3.amplitude = 0.2
        layer3.harmonics = 1
        layer3.mod_freq = 15
        layer3.mod_depth = 0.4
        layer3.active = True
        self.layers.append(layer3)
        self.layer_counter = 3
        
        self.refresh_layers_ui()
        self.status_label.text = 'Loaded: Water preset'
    
    def load_metal_preset(self, instance):
        self.current_preset = 'metal'
        self.highlight_preset_button('metal')
        
        # Clear existing layers
        self.clear_layers()
        
        # Layer 1: Sharp impact
        layer1 = SoundLayer(1)
        layer1.waveform = 'square'
        layer1.frequency = 600
        layer1.amplitude = 0.8
        layer1.harmonics = 4
        layer1.mod_freq = 0
        layer1.mod_depth = 0
        layer1.active = True
        self.layers.append(layer1)
        self.layer_counter = 1
        
        # Layer 2: Resonant ringing
        layer2 = SoundLayer(2)
        layer2.waveform = 'sine'
        layer2.frequency = 800
        layer2.amplitude = 0.5
        layer2.harmonics = 6
        layer2.mod_freq = 12
        layer2.mod_depth = 0.6
        layer2.active = True
        self.layers.append(layer2)
        self.layer_counter = 2
        
        # Layer 3: Low metallic thud
        layer3 = SoundLayer(3)
        layer3.waveform = 'triangle'
        layer3.frequency = 100
        layer3.amplitude = 0.4
        layer3.harmonics = 2
        layer3.mod_freq = 0
        layer3.mod_depth = 0
        layer3.active = True
        self.layers.append(layer3)
        self.layer_counter = 3
        
        self.refresh_layers_ui()
        self.status_label.text = 'Loaded: Metal impact preset'
    
    def load_train_preset(self, instance):
        self.current_preset = 'train'
        self.highlight_preset_button('train')
        
        # Clear existing layers
        self.clear_layers()
        
        # Layer 1: Low rumble
        layer1 = SoundLayer(1)
        layer1.waveform = 'sine'
        layer1.frequency = 80
        layer1.amplitude = 0.6
        layer1.harmonics = 3
        layer1.mod_freq = 2
        layer1.mod_depth = 0.3
        layer1.active = True
        self.layers.append(layer1)
        self.layer_counter = 1
        
        # Layer 2: Wheel noise
        layer2 = SoundLayer(2)
        layer2.waveform = 'noise'
        layer2.frequency = 300
        layer2.amplitude = 0.3
        layer2.harmonics = 0
        layer2.mod_freq = 10
        layer2.mod_depth = 0.4
        layer2.active = True
        self.layers.append(layer2)
        self.layer_counter = 2
        
        # Layer 3: Whistle (if needed)
        layer3 = SoundLayer(3)
        layer3.waveform = 'sine'
        layer3.frequency = 1000
        layer3.amplitude = 0.2
        layer3.harmonics = 1
        layer3.mod_freq = 0
        layer3.mod_depth = 0
        layer3.active = True
        self.layers.append(layer3)
        self.layer_counter = 3
        
        self.refresh_layers_ui()
        self.status_label.text = 'Loaded: Train preset'
    
    def clear_layers(self):
        self.layers.clear()
        self.layer_counter = 0
        self.layers_layout.clear_widgets()
    
    def refresh_layers_ui(self):
        self.layers_layout.clear_widgets()
        for layer in self.layers:
            layer_control = LayerControl(layer, self.on_layer_change)
            layer_control.children[-1].children[0].bind(
                on_press=lambda x, lc=layer_control: self.delete_layer(lc)
            )
            self.layers_layout.add_widget(layer_control)
        self.on_layer_change()
    
    def play_current_preset(self, instance):
        if not self.current_preset:
            self.show_popup('Error', 'Please load a preset first')
            return
            
        self.status_label.text = f'Playing {self.current_preset} preset...'
        Clock.schedule_once(self._play_sound, 0.1)
    
    def play_sound(self, instance):
        self.status_label.text = 'Generating custom sound...'
        Clock.schedule_once(self._play_sound, 0.1)
    
    def _play_sound(self, dt):
        try:
            # Stop any currently playing sound
            if self.current_sound:
                self.current_sound.stop()
            
            # Generate mixed audio data
            audio_data = self.generator.generate_mixed_sound(self.layers)
            
            # Clean up previous temporary file
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            self.temp_file_path = temp_file.name
            temp_file.close()
            
            # Save audio data to file
            try:
                from scipy.io.wavfile import write
                # Convert to 16-bit integers for WAV format
                scaled_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
                write(self.temp_file_path, self.generator.sample_rate, scaled_data)
            except ImportError:
                # Fallback if scipy is not available
                import wave
                with wave.open(self.temp_file_path, 'w') as wav_file:
                    wav_file.setparams((1, 2, self.generator.sample_rate, 0, 'NONE', 'not compressed'))
                    scaled_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
                    wav_file.writeframes(scaled_data.tobytes())
            
            # Load and play sound
            self.current_sound = SoundLoader.load(self.temp_file_path)
            if self.current_sound:
                self.current_sound.play()
                if self.current_preset:
                    self.status_label.text = f'Playing {self.current_preset} preset...'
                else:
                    self.status_label.text = 'Playing custom sound...'
            else:
                self.status_label.text = 'Error: Could not load sound'
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
    
    def save_sound(self, instance):
        self.status_label.text = 'Saving sound...'
        Clock.schedule_once(self._save_sound, 0.1)
    
    def _save_sound(self, dt):
        try:
            # Generate mixed audio data
            audio_data = self.generator.generate_mixed_sound(self.layers)
            
            # Create save file path
            save_path = os.path.join(os.getcwd(), 'sfx_output.wav')
            
            # Save audio data
            try:
                from scipy.io.wavfile import write
                # Convert to 16-bit integers for WAV format
                scaled_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
                write(save_path, self.generator.sample_rate, scaled_data)
                self.status_label.text = f'Saved: {save_path}'
                self.show_popup('Success', f'Sound saved to:\n{save_path}')
            except ImportError:
                self.status_label.text = 'Error: scipy required for saving'
                self.show_popup('Error', 'scipy required for saving files')
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'

if __name__ == '__main__':
    SFXApp().run()