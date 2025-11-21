function indexCopyButtons() {
  const clipButtons = document.querySelectorAll(".clip-button");

  clipButtons.forEach((button, index) => {
    button.classList.add(`copy-btn-${index}`);
  });

  return clipButtons.length;
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", indexCopyButtons);
} else {
  indexCopyButtons();
}
