const frame = document.getElementById("hero-frame");
const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
let heroVideo;

function syncHero() {
  if (motion.matches) {
    if (heroVideo) {
      heroVideo.pause();
      heroVideo.remove();
      heroVideo = null;
    }
    return;
  }
  if (!heroVideo) {
    heroVideo = document.createElement("video");
    heroVideo.src = "media/hero.mp4";
    heroVideo.muted = true;
    heroVideo.loop = true;
    heroVideo.autoplay = true;
    heroVideo.playsInline = true;
    heroVideo.setAttribute("aria-hidden", "true");
    frame.append(heroVideo);
    heroVideo.play().catch(() => {
      heroVideo.remove();
      heroVideo = null;
    });
  }
}

syncHero();
motion.addEventListener("change", syncHero);
document.addEventListener("visibilitychange", () => {
  if (!heroVideo) return;
  if (document.hidden) heroVideo.pause();
  else if (!motion.matches) heroVideo.play().catch(() => {});
});
