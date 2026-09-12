/* Guidebook behaviour. Two things only: the mobile nav toggle, and marking the
   current page in the nav. Everything else is static HTML. */

(function () {
  'use strict';

  var toggle = document.querySelector('.nav-toggle');
  var nav = document.querySelector('.site-nav');

  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  /* The nav lives in every page's markup, so the current entry is found by
     comparing file names rather than hard-coding aria-current per page. */
  var here = window.location.pathname.split('/').pop() || 'index.html';
  var links = document.querySelectorAll('.site-nav a');
  for (var i = 0; i < links.length; i += 1) {
    var href = links[i].getAttribute('href') || '';
    if (href === here || (here === 'index.html' && href === './')) {
      links[i].setAttribute('aria-current', 'page');
    }
  }
})();
