const field = {
    c : ".console",
    exe : ".console.exe"
};

function _translate(code) {
    document.getElementsByTagName("html")[0].lang = code;
    for(let i = 0; i < strArr.length; i++) {
        document.getElementById(strArr[i].id).innerHTML = strArr[i][code];
    }
}

function endl(count = 1) {
    return '\n'.repeat(count);
}

document.write('<p class="indent" style="color:rgb(95, 95, 95); font-size: 10px;">© 2023. YuiSanae2f</p>');
document.getElementById("_t").innerHTML = `<h2 id="translate"></h2><div class="box indent"><d onclick="_translate('ko')">한국어</d><br><d onclick="_translate('en')">English</d><br></div>`;

// 현재 페이지의 URL에서 쿼리 매개변수 가져오기
const queryString = window.location.search;

// URLSearchParams 객체를 사용하여 쿼리 매개변수 파싱
const urlParams = new URLSearchParams(queryString);

// "q" 매개변수의 값을 가져오기
const langCode = urlParams.get('lang');
if(langCode == null) langCode = "ko";

_translate(langCode);
aEl = document.getElementsByTagName('a');
for(let i = 0; i < aEl.length; i++) {
    aEl[i].href += `?lang=${langCode}`;
}

// © 2023. YuiSanae2f