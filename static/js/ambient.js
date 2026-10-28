/* ================================================================
   EcoPrice — Ambient Sound System
   Inspired by Opera Browser's background sounds.
   Supports custom audio file OR Web Audio API generated tones.
   Persists playback state & position across page navigation.
   ================================================================ */

const AmbientSound = {
    audioEl: null,
    ctx: null,
    isPlaying: false,
    gainNode: null,
    oscillators: [],
    volume: 0.15,
    useFile: false,

    init() {
        // Retrieve persisted volume if available
        var savedVol = localStorage.getItem('ecoprice-ambient-volume');
        if (savedVol !== null) {
            this.volume = parseFloat(savedVol);
        }

        // Sync any volume slider on page
        var self = this;
        document.querySelectorAll('.volume-slider').forEach(function(slider) {
            slider.value = Math.round(self.volume * 100);
        });

        // Check if a custom audio file exists
        var audio = document.getElementById('ambient-audio');
        if (audio) {
            this.audioEl = audio;
            this.audioEl.volume = this.volume;
            this.audioEl.loop = true;
            this.useFile = true;

            // Restore saved playback position
            var savedTime = localStorage.getItem('ecoprice-ambient-time');
            if (savedTime) {
                var t = parseFloat(savedTime) || 0;
                if (!isNaN(t) && t > 0) {
                    try {
                        this.audioEl.currentTime = t;
                    } catch(e) {}
                }
            }

            // Continuously track playback time
            this.audioEl.addEventListener('timeupdate', function() {
                localStorage.setItem('ecoprice-ambient-time', self.audioEl.currentTime);
            });

            // Save state on unload
            window.addEventListener('beforeunload', function() {
                if (self.audioEl) {
                    localStorage.setItem('ecoprice-ambient-time', self.audioEl.currentTime);
                }
            });

            return;
        }

        // Fallback to Web Audio API
        if (this.ctx) return;
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.gainNode = this.ctx.createGain();
        this.gainNode.gain.value = 0;
        this.gainNode.connect(this.ctx.destination);
    },

    play() {
        this.init();

        // Persist playing state
        localStorage.setItem('ecoprice-ambient-playing', 'true');
        var self = this;

        if (this.useFile && this.audioEl) {
            this.audioEl.volume = this.volume;
            var playPromise = this.audioEl.play();
            if (playPromise !== undefined) {
                playPromise.then(function() {
                    self.isPlaying = true;
                    self.updateButtons(true);
                }).catch(function(err) {
                    // Browser autoplay policy might restrict playback without a direct gesture on this doc
                    self.updateButtons(true);
                    function resumeOnGesture() {
                        if (localStorage.getItem('ecoprice-ambient-playing') === 'true') {
                            self.audioEl.play().then(function() {
                                self.isPlaying = true;
                                self.updateButtons(true);
                            }).catch(function() {});
                        }
                        document.removeEventListener('click', resumeOnGesture);
                        document.removeEventListener('keydown', resumeOnGesture);
                        document.removeEventListener('touchstart', resumeOnGesture);
                    }
                    document.addEventListener('click', resumeOnGesture);
                    document.addEventListener('keydown', resumeOnGesture);
                    document.addEventListener('touchstart', resumeOnGesture);
                });
            }
            return;
        }

        // Web Audio API fallback — soft ambient drone
        if (!this.ctx) return;
        var freqs = [65, 98, 131, 196];
        freqs.forEach(function(freq, i) {
            var osc = self.ctx.createOscillator();
            var oscGain = self.ctx.createGain();

            osc.type = 'sine';
            osc.frequency.value = freq;
            osc.detune.value = Math.random() * 6 - 3;

            oscGain.gain.value = 0.04 - (i * 0.005);
            osc.connect(oscGain);
            oscGain.connect(self.gainNode);
            osc.start();

            self.oscillators.push({ osc: osc, gain: oscGain });
        });

        this.gainNode.gain.setTargetAtTime(this.volume, this.ctx.currentTime, 1.5);
        this.isPlaying = true;
        this.updateButtons(true);
    },

    stop() {
        this.isPlaying = false;
        localStorage.setItem('ecoprice-ambient-playing', 'false');
        this.updateButtons(false);

        if (this.useFile && this.audioEl) {
            this.audioEl.pause();
            return;
        }

        if (!this.ctx) return;
        this.gainNode.gain.setTargetAtTime(0, this.ctx.currentTime, 0.8);

        var self = this;
        setTimeout(function() {
            self.oscillators.forEach(function(o) {
                try { o.osc.stop(); } catch(e) {}
            });
            self.oscillators = [];
        }, 2000);
    },

    toggle() {
        if (this.isPlaying) {
            this.stop();
            return false;
        } else {
            this.play();
            return true;
        }
    },

    setVolume(v) {
        this.volume = Math.max(0, Math.min(1, v));
        localStorage.setItem('ecoprice-ambient-volume', this.volume);
        if (this.useFile && this.audioEl) {
            this.audioEl.volume = this.volume;
        } else if (this.isPlaying && this.gainNode) {
            this.gainNode.gain.setTargetAtTime(this.volume, this.ctx.currentTime, 0.1);
        }
    },

    updateButtons(active) {
        document.querySelectorAll('.btn-ambient').forEach(function(btn) {
            if (active) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
};

/* ── UI Click Sounds ── */
var UISound = {
    ctx: null,

    init: function() {
        if (this.ctx) return;
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    },

    /* Soft click for button presses */
    click: function() {
        this.init();
        var osc = this.ctx.createOscillator();
        var gain = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.value = 800;
        gain.gain.value = 0.08;
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.08);

        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.08);
    },

    /* Soft success tone */
    success: function() {
        this.init();
        var notes = [523, 659, 784];
        var self = this;
        notes.forEach(function(freq, i) {
            var osc = self.ctx.createOscillator();
            var gain = self.ctx.createGain();

            osc.type = 'sine';
            osc.frequency.value = freq;
            gain.gain.value = 0.06;
            gain.gain.exponentialRampToValueAtTime(0.001, self.ctx.currentTime + 0.15 + i * 0.08);

            osc.connect(gain);
            gain.connect(self.ctx.destination);
            osc.start(self.ctx.currentTime + i * 0.08);
            osc.stop(self.ctx.currentTime + 0.25 + i * 0.08);
        });
    },

    /* Hover tick — very subtle */
    hover: function() {
        this.init();
        var osc = this.ctx.createOscillator();
        var gain = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.value = 1200;
        gain.gain.value = 0.03;
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.03);

        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.03);
    }
};

/* ── Auto-attach sounds & restore persistent ambient state ── */
document.addEventListener('DOMContentLoaded', function() {
    // 1. Check persistent ambient playing state
    AmbientSound.init();
    var wasPlaying = localStorage.getItem('ecoprice-ambient-playing') === 'true';
    if (wasPlaying) {
        AmbientSound.play();
    } else {
        AmbientSound.updateButtons(false);
    }

    // 2. Click sounds on interactive elements
    var clickSelectors = [
        '.btn-action', '.btn-landing', '.btn-auth', '.btn-google',
        '.dash-nav-item', '.theme-toggle-btn', '.btn-expand',
        '.btn-ambient', '.landing-nav a', '.btn-page-action',
        '.filter-pill', '.back-link', '.toggle-switch-sm',
        '.team-mail-link', '.landing-footer-link'
    ].join(', ');

    document.querySelectorAll(clickSelectors).forEach(function(el) {
        el.addEventListener('click', function() { UISound.click(); });
    });

    // 3. Hover ticks on nav items
    document.querySelectorAll('.dash-nav-item, .landing-nav a').forEach(function(el) {
        el.addEventListener('mouseenter', function() { UISound.hover(); });
    });
});
