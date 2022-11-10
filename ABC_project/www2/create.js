//=========================================================================
function gid(id) {return document.getElementById(id);}
//=========================================================================

//=========================================================================
//열을 추가하는 함수
function appendRow(bread, count, listName, sum) {
    //tbodyToDo를 이용하여 새 행을 생성한다s
    let tbody = gid(listName);
    let newRow = tbody.insertRow();
    //생성된 새 행(newRow)을 이용하여 내부에 두개의 행을 생성한다.
    let cell0 = newRow.insertCell(0);
    let cell1 = newRow.insertCell(1);
    let cell2 = newRow.insertCell(2);
    let cell3 = newRow.insertCell(3);
    let cell4 = newRow.insertCell(4);
    let cell5 = newRow.insertCell(5);
    let cell6 = newRow.insertCell(6);

    count = parseInt(count)
    let price = 0;
    if(bread == 'chocoPie') {
        bread = '초코파이';
        price = 2000;
    }
    else if(bread == 'custard') {
        bread = '카스타드';
        price = 4000;
    }
    else if(bread == 'croissant') {
        bread = '크로와상';
        price = 1000;
    }
    else if(bread == 'pie') {
        bread = '파이'
        price = 1500;
    }
    else if(bread == 'pizzaBread') {
        bread = '피자빵';
        price = 2500;
    }
    else if(bread == 'stickBread') {
        bread = '버터스틱';
        price = 1000;
    }
    
    //생성된 셀에 필요한 내용을 저장한다.
    cell0.innerHTML = "<strong>" + bread + "</strong>";
    cell1.innerHTML = "<strong>" + price + "</strong>"; 
    cell2.innerHTML = "<Strong>" + count + "</Strong>";
    cell3.innerHTML = "<Strong>" + "+" + "</Strong>";
    cell4.innerHTML = "<Strong>" + "-" + "</Strong>";
    cell5.innerHTML = "<Strong id = 'count'>" + count * price + "</Strong>";
    cell6.innerHTML = "<Strong>" + 'X' + "</Strong>";
    //생성된 셀에 필요한 이벤트 핸들러를 저장한다.
    $(cell3).click(btnPlus);
    $(cell4).click(btnMinus);
    $(cell6).click(btnDeleteHandler);

    sum += count * price
    return sum;
}
//==========================================================================
//--------------------------------------------------------------------------
//삭제 버튼
function btnDeleteHandler(){
    //전체 가격 가져오기
    let a = $('#resum').text();
    let before_sum = Number(a);
    //합계 가져오기
    let b = this.parentNode.children[5].children[0].innerHTML;
    let res = Number(b);

    //전체 가격 빼주기
    let sum = gid('resum');
    sum.innerHTML = before_sum - res;
    //셀 삭제해주기
    $(this.parentNode).remove(); 
}
//--------------------------------------------------------------------------
//플러스 버튼
function btnPlus(){
    //갯수 가져오기
    let a = this.parentNode.children[2].children[0].innerHTML;
    let count = Number(a);
    //가격 가져오기
    let b = this.parentNode.children[1].children[0].innerHTML;
    let price = Number(b);
    //전체 가격 가져오기
    let c = $('#resum').text();
    let before_sum = Number(c);
    let after_sum = before_sum + price;
    //갯수 추가하기
    count += 1;
    //각각 맞는값 붙여주기 
    this.parentNode.children[2].children[0].innerHTML = count;
    this.parentNode.children[5].children[0].innerHTML = count * price;
    let sum = gid('resum');
    sum.innerHTML = after_sum;
}
//--------------------------------------------------------------------------
//마이너스 버튼
function btnMinus(){
    //갯수 가져오기
    let a = this.parentNode.children[2].children[0].innerHTML;
    let count = Number(a);
    //가격 가져오기
    let b = this.parentNode.children[1].children[0].innerHTML;
    let price = Number(b);
    //전체 가격 가져오기
    let c = $('#resum').text();
    let before_sum = Number(c);
    let after_sum = before_sum - price;
    //갯수 추가하기
    count -= 1;
    //각각 맞는값 붙여주기 
    this.parentNode.children[2].children[0].innerHTML = count;
    this.parentNode.children[5].children[0].innerHTML = count * price;
    let sum = gid('resum');
    sum.innerHTML = after_sum;

}
//--------------------------------------------------------------------------
//단어 끊어주기
function words(str, sum) {
    let words = str.split(',');
    let res = new Array(words.length);
    let j = 0;
    for(let i = 0; i < words.length; i++){
        res[i] = words[i].split(' ');
    }

    for(let i = 0; i < words.length; i++){
        sum = appendRow(res[i][0], res[i][1], 'ordertable', sum)
    } 
    return sum;
    
}

//--------------------------------------------------------------------------

//detect.py 파일 재가동
async function back(type){
    const resultElement = document.getElementById('goback');
    if(type === 'goback'){  
        window.close('new.html');
        await eel.reopen();

        //새로 detect 한 데이터를 eel을 이용하여 html에 전송
    }
}

// 판매내역 조회 팝업창
function showList(){
   
    window.open('data.html',"a","width=800, height=800, top=100, left=550");
      
}

//계산 진행시 영수증 팝업창
function showPopup(){

    window.open('receipt.html',"a","width=800, height=800, top=100, left=550");
    document.getElementById('ordertable').innerHTML;

    let bb = (document.getElementById('ordertable').innerHTML);
    console.log(bb);

    localStorage.setItem('bb',bb);
}

