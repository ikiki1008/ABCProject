//=========================================================================
function gid(id) {return document.getElementById(id);}
//=========================================================================

//=========================================================================
//열을 추가하는 함수
function appendRow(bread, count, listName) {
    //tbodyToDo를 이용하여 새 행을 생성한다s
    let tbody = gid(listName);
    let newRow = tbody.insertRow();
    //생성된 새 행(newRow)을 이용하여 내부에 두개의 행을 생성한다.
    let cell0 = newRow.insertCell(0);
    let cell1 = newRow.insertCell(1);
    let cell2 = newRow.insertCell(2);
    let cell3 = newRow.insertCell(3);

    count = parseInt(count)
    let price = 0;
    if(bread == '초코파이') {
        price = 2000;
    }
    else if(bread == '카스타드') {
        price = 4000;
    }
    else if(bread == '크로와상') {
        price = 1000;
    }
    else if(bread == '파이') {
        price = 1500;
    }
    else if(bread == '피자빵') {
        price = 2500;
    }
    else if(bread == '버터스틱') {
        price = 1000;
    }
    
    //생성된 셀에 필요한 내용을 저장한다.
    cell0.innerHTML = "<strong>" + bread + "</strong>";
    cell1.innerHTML = "<strong>" + price + "</strong>"; 
    cell2.innerHTML = "<Strong>" + count + "</Strong>";
    cell3.innerHTML = "<Strong id = 'count'>" + count * price + "</Strong>";

    return count * price

}
//==========================================================================
//--------------------------------------------------------------------------
//단어 끊어주기
function words(str) {
    let words = str.split(',');
    console.log(words)
    let res = new Array(words.length);
    let sum = 0;
    let j = 0;
    for(let i = 0; i < words.length; i++){
        if(i % 7 == 0){
            res[j] += words[i];
            res[j] += ' ';
            res[j] += words[i+2];
            res[j] = res[j].replace("undefined","");
            j++;
        }
    }

    for(let i = 0; i < words.length/7-1; i++){ 
        if (typeof res[i] != "undefined") {  // Checking if the variable is defined.
            res[i] = res[i].split(' ');
        }
        console.log(i, res[i])
    }
    console.log(words.length/7);
    for(let i = 0; i < words.length/7-1; i++){
        if(res[i][1] != 'NaN') {
            sum += appendRow(res[i][0], res[i][1], 'ordertable')
        }
    } 
    console.log(sum)
    gid("total").innerHTML += sum;

    
}