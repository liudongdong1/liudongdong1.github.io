/**
 * Custom javascript for FixIt site.
 * @author @Kendrick https://kendrickcheung.github.io/
 */
const Docs = new (function () {
  /**
   * Rest in Peace. R.I.P 🕯️
   * @2022-3-28 [3.21-mu5735] 沉痛哀悼 132 名遇难同胞：东航航班失事，遇难者含旅客 123 人，机组 9 人
   * @returns {Docs}
   */
  this.RIP = () => {
    if (new Date() < new Date('2022-03-31')) {
      document.querySelector('html').style.filter = 'grayscale(100%)';
    }
    return this;
  };
  /**
   * Initialize.
   * @returns {Docs}
   */
  this.init = () => {
    this.RIP();
    return this;
  };
})();

/**
 * Immediate.
 */
(() => {
  // Docs.init();
  // It will be executed when the DOM tree is built.
  document.addEventListener('DOMContentLoaded', () => {
    // Docs.init();
  });
})();

// 点击显示隐藏文本
function hideTextClick() {
  if (this.querySelector('.hide').style.display === 'inline') {
    this.querySelector('.check').style.display = 'inline-block';
    this.querySelector('.hide').style.display = 'none'
  } else {
    this.querySelector('.check').style.display = 'none';
    this.querySelector('.hide').style.display = 'inline'
  }

}
var hideTextList = document.getElementsByClassName('hide-text');
for (var i in hideTextList) {
  hideTextList[i].onclick = hideTextClick;
}

var blankTextList = document.getElementsByClassName('blank-text');
for (var i in blankTextList) {
  blankTextList[i].onclick = hideTextClick;
}