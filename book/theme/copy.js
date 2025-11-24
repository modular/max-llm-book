function indexCopyButtons() {
    const clipButtons = document.querySelectorAll('.clip-button');
    clipButtons.forEach((button, index) => {
      button.classList.add(`copy-btn-${index}`);
    });
  }
  
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', indexCopyButtons);
  } else {
    indexCopyButtons();
  }