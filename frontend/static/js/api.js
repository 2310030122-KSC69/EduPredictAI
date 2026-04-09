// api.js - Shared utilities

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// Close sidebar on outside click (mobile)
document.addEventListener('click', function(e) {
  const sb = document.getElementById('sidebar');
  const btn = document.querySelector('.menu-btn');
  if (sb && sb.classList.contains('open') && !sb.contains(e.target) && e.target !== btn) {
    sb.classList.remove('open');
  }
});

function cleanInput(val){
  return val.trim().replace(/\s+/g, " ");
}